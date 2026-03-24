use anyhow::{Result, bail};
use candle_core::{D, Device, Tensor};
use candle_nn::{Embedding, Init, Linear, Module, VarBuilder, VarMap};

use crate::config::Hyperparameters;

pub const RMS_NORM_EPS: f64 = 1.0e-5;

#[derive(Clone)]
pub struct LinearNoBias {
    inner: Linear,
}

impl LinearNoBias {
    fn new(vb: VarBuilder<'_>, in_dim: usize, out_dim: usize, init: Init) -> Result<Self> {
        let weight = vb.get_with_hints((out_dim, in_dim), "weight", init)?;
        Ok(Self {
            inner: Linear::new(weight, None),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.inner.forward(x)?)
    }
}

#[derive(Clone)]
pub struct CausalSelfAttention {
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    c_q: LinearNoBias,
    c_k: LinearNoBias,
    c_v: LinearNoBias,
    proj: LinearNoBias,
    q_gain: Tensor,
    rope_base: f64,
}

impl CausalSelfAttention {
    fn new(vb: VarBuilder<'_>, cfg: &Hyperparameters) -> Result<Self> {
        let head_dim = cfg.model_dim / cfg.num_heads;
        Ok(Self {
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim,
            c_q: LinearNoBias::new(
                vb.pp("c_q"),
                cfg.model_dim,
                cfg.model_dim,
                candle_nn::init::DEFAULT_KAIMING_NORMAL,
            )?,
            c_k: LinearNoBias::new(
                vb.pp("c_k"),
                cfg.model_dim,
                cfg.num_kv_heads * head_dim,
                candle_nn::init::DEFAULT_KAIMING_NORMAL,
            )?,
            c_v: LinearNoBias::new(
                vb.pp("c_v"),
                cfg.model_dim,
                cfg.num_kv_heads * head_dim,
                candle_nn::init::DEFAULT_KAIMING_NORMAL,
            )?,
            proj: LinearNoBias::new(
                vb.pp("proj"),
                cfg.model_dim,
                cfg.model_dim,
                Init::Const(0.0),
            )?,
            q_gain: vb.get_with_hints(cfg.num_heads, "q_gain", Init::Const(cfg.qk_gain_init))?,
            rope_base: cfg.rope_base,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (bsz, seqlen, dim) = x.dims3()?;
        let q = self
            .c_q
            .forward(x)?
            .reshape((bsz, seqlen, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .c_k
            .forward(x)?
            .reshape((bsz, seqlen, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .c_v
            .forward(x)?
            .reshape((bsz, seqlen, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = rms_norm_no_weight(&q, RMS_NORM_EPS)?;
        let k = rms_norm_no_weight(&k, RMS_NORM_EPS)?;
        let (cos, sin) = rotary_tables(seqlen, self.head_dim, self.rope_base, x.device())?;
        let q = apply_rotary_emb(&q, &cos, &sin)?;
        let k = apply_rotary_emb(&k, &cos, &sin)?;
        let q = q.broadcast_mul(&self.q_gain.reshape((1, self.num_heads, 1, 1))?)?;

        let (k, v) = repeat_kv_for_gqa(&k, &v, self.num_heads, self.num_kv_heads)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let att = q.matmul(&k.transpose(2, 3)?)?.affine(scale, 0.0)?;
        let att = att.broadcast_add(&causal_mask(seqlen, x.device())?)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((bsz, seqlen, dim))?;
        self.proj.forward(&y)
    }
}

#[derive(Clone)]
pub struct Mlp {
    fc: LinearNoBias,
    proj: LinearNoBias,
}

impl Mlp {
    fn new(vb: VarBuilder<'_>, cfg: &Hyperparameters) -> Result<Self> {
        let hidden = cfg.model_dim * cfg.mlp_mult;
        Ok(Self {
            fc: LinearNoBias::new(
                vb.pp("fc"),
                cfg.model_dim,
                hidden,
                candle_nn::init::DEFAULT_KAIMING_NORMAL,
            )?,
            proj: LinearNoBias::new(vb.pp("proj"), hidden, cfg.model_dim, Init::Const(0.0))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc.forward(x)?.relu()?.sqr()?;
        self.proj.forward(&x)
    }
}

#[derive(Clone)]
pub struct Block {
    attn: CausalSelfAttention,
    mlp: Mlp,
    attn_scale: Tensor,
    mlp_scale: Tensor,
    resid_mix: Tensor,
}

impl Block {
    fn new(vb: VarBuilder<'_>, cfg: &Hyperparameters) -> Result<Self> {
        let resid_mix = vb.get_with_hints((2, cfg.model_dim), "resid_mix", Init::Const(0.0))?;
        Ok(Self {
            attn: CausalSelfAttention::new(vb.pp("attn"), cfg)?,
            mlp: Mlp::new(vb.pp("mlp"), cfg)?,
            attn_scale: vb.get_with_hints(cfg.model_dim, "attn_scale", Init::Const(1.0))?,
            mlp_scale: vb.get_with_hints(cfg.model_dim, "mlp_scale", Init::Const(1.0))?,
            resid_mix,
        })
    }

    fn forward(&self, x: &Tensor, x0: &Tensor) -> Result<Tensor> {
        let mix = self.resid_mix.broadcast_as((1, 1, 2, x.dim(D::Minus1)?))?;
        let mix0 = mix.narrow(2, 0, 1)?.squeeze(2)?;
        let mix1 = mix.narrow(2, 1, 1)?.squeeze(2)?;
        let x = mix0
            .broadcast_mul(x)?
            .broadcast_add(&mix1.broadcast_mul(x0)?)?;
        let attn_out = self.attn.forward(&rms_norm_no_weight(&x, RMS_NORM_EPS)?)?;
        let x = x.broadcast_add(
            &self
                .attn_scale
                .reshape((1, 1, ()))?
                .broadcast_mul(&attn_out)?,
        )?;
        let mlp_out = self.mlp.forward(&rms_norm_no_weight(&x, RMS_NORM_EPS)?)?;
        Ok(x.broadcast_add(
            &self
                .mlp_scale
                .reshape((1, 1, ()))?
                .broadcast_mul(&mlp_out)?,
        )?)
    }
}

#[derive(Clone)]
pub struct Gpt {
    pub varmap: VarMap,
    tok_emb: Embedding,
    blocks: Vec<Block>,
    skip_weights: Tensor,
    lm_head: Option<LinearNoBias>,
    tie_embeddings: bool,
    logit_softcap: f64,
    num_encoder_layers: usize,
    num_decoder_layers: usize,
}

impl Gpt {
    pub fn new(cfg: &Hyperparameters, device: &Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);
        let tok_emb_weight = vb.pp("tok_emb").get_with_hints(
            (cfg.vocab_size, cfg.model_dim),
            "weight",
            Init::Randn {
                mean: 0.0,
                stdev: cfg.tied_embed_init_std,
            },
        )?;
        let tok_emb = Embedding::new(tok_emb_weight, cfg.model_dim);
        let num_encoder_layers = cfg.num_layers / 2;
        let num_decoder_layers = cfg.num_layers - num_encoder_layers;
        let blocks = (0..cfg.num_layers)
            .map(|index| Block::new(vb.pp("blocks").pp(index), cfg))
            .collect::<Result<Vec<_>>>()?;
        let skip_weights = vb.get_with_hints(
            (num_encoder_layers.min(num_decoder_layers), cfg.model_dim),
            "skip_weights",
            Init::Const(1.0),
        )?;
        let lm_head = if cfg.tie_embeddings {
            None
        } else {
            Some(LinearNoBias::new(
                vb.pp("lm_head"),
                cfg.model_dim,
                cfg.vocab_size,
                Init::Const(0.0),
            )?)
        };
        initialize_resid_mix(
            &varmap,
            num_encoder_layers + num_decoder_layers,
            cfg.model_dim,
            device,
        )?;
        Ok(Self {
            varmap,
            tok_emb,
            blocks,
            skip_weights,
            lm_head,
            tie_embeddings: cfg.tie_embeddings,
            logit_softcap: cfg.logit_softcap,
            num_encoder_layers,
            num_decoder_layers,
        })
    }

    pub fn forward_loss(&self, input_ids: &Tensor, target_ids: &Tensor) -> Result<Tensor> {
        let mut x = self.tok_emb.forward(input_ids)?;
        x = rms_norm_no_weight(&x, RMS_NORM_EPS)?;
        let x0 = x.clone();
        let mut skips = Vec::with_capacity(self.num_encoder_layers);

        for index in 0..self.num_encoder_layers {
            x = self.blocks[index].forward(&x, &x0)?;
            skips.push(x.clone());
        }
        for index in 0..self.num_decoder_layers {
            if let Some(skip) = skips.pop() {
                let weight =
                    self.skip_weights
                        .narrow(0, index, 1)?
                        .reshape((1, 1, x.dim(D::Minus1)?))?;
                x = x.broadcast_add(&weight.broadcast_mul(&skip)?)?;
            }
            x = self.blocks[self.num_encoder_layers + index].forward(&x, &x0)?;
        }

        let x = rms_norm_no_weight(&x, RMS_NORM_EPS)?.reshape(((), x.dim(D::Minus1)?))?;
        let targets = target_ids.flatten_all()?;
        let logits_proj = if self.tie_embeddings {
            x.matmul(&self.tok_emb.embeddings().t()?)?
        } else {
            self.lm_head
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("lm_head missing"))?
                .forward(&x)?
        };
        let logits = logits_proj
            .affine(1.0 / self.logit_softcap, 0.0)?
            .tanh()?
            .affine(self.logit_softcap, 0.0)?;
        Ok(candle_nn::loss::cross_entropy(&logits, &targets)?)
    }
}

pub fn rms_norm_no_weight(x: &Tensor, eps: f64) -> Result<Tensor> {
    let rms = x.sqr()?.mean_keepdim(D::Minus1)?.affine(1.0, eps)?.sqrt()?;
    Ok(x.broadcast_div(&rms)?)
}

fn rotary_tables(
    seq_len: usize,
    head_dim: usize,
    base: f64,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(seq_len * half);
    let mut sin = Vec::with_capacity(seq_len * half);
    for pos in 0..seq_len {
        for i in 0..half {
            let inv_freq = 1.0 / base.powf((2 * i) as f64 / head_dim as f64);
            let angle = pos as f64 * inv_freq;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    Ok((
        Tensor::from_vec(cos, (1, 1, seq_len, half), device)?,
        Tensor::from_vec(sin, (1, 1, seq_len, half), device)?,
    ))
}

fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let left = x1
        .broadcast_mul(cos)?
        .broadcast_add(&x2.broadcast_mul(sin)?)?;
    let right = x1
        .broadcast_mul(&sin.affine(-1.0, 0.0)?)?
        .broadcast_add(&x2.broadcast_mul(cos)?)?;
    Ok(Tensor::cat(&[left, right], D::Minus1)?)
}

fn repeat_kv_for_gqa(
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
) -> Result<(Tensor, Tensor)> {
    if num_heads == num_kv_heads {
        return Ok((k.clone(), v.clone()));
    }
    if num_heads % num_kv_heads != 0 {
        bail!("num_heads must be divisible by num_kv_heads");
    }
    let groups = num_heads / num_kv_heads;
    let (bsz, _, seqlen, head_dim) = k.dims4()?;
    let k = k
        .reshape((bsz, num_kv_heads, 1, seqlen, head_dim))?
        .broadcast_as((bsz, num_kv_heads, groups, seqlen, head_dim))?
        .reshape((bsz, num_heads, seqlen, head_dim))?;
    let (bsz, _, seqlen, head_dim) = v.dims4()?;
    let v = v
        .reshape((bsz, num_kv_heads, 1, seqlen, head_dim))?
        .broadcast_as((bsz, num_kv_heads, groups, seqlen, head_dim))?
        .reshape((bsz, num_heads, seqlen, head_dim))?;
    Ok((k, v))
}

fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mut mask = Vec::with_capacity(seq_len * seq_len);
    for q in 0..seq_len {
        for k in 0..seq_len {
            mask.push(if k > q { -1.0e9_f32 } else { 0.0_f32 });
        }
    }
    Ok(Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)?)
}

fn initialize_resid_mix(
    varmap: &VarMap,
    num_layers: usize,
    dim: usize,
    device: &Device,
) -> Result<()> {
    let mut lock = varmap.data().lock().unwrap();
    for layer in 0..num_layers {
        let key = format!("blocks.{layer}.resid_mix");
        let Some(var) = lock.get_mut(&key) else {
            continue;
        };
        let mut values = vec![0_f32; 2 * dim];
        values[..dim].fill(1.0);
        let tensor = Tensor::from_vec(values, (2, dim), device)?;
        var.set(&tensor)?;
    }
    Ok(())
}
