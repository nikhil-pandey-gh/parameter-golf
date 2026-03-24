use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, anyhow, bail};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};

use crate::config::Hyperparameters;
use crate::data::{DistributedTokenLoader, load_validation_tokens};
use crate::model::Gpt;
use crate::quant::{
    QuantStats, load_quantized_state, quantize_varmap, restore_quantized_state,
    save_quantized_state,
};
use crate::sentencepiece::{TokenMetricLuts, build_sentencepiece_luts};

const CONTROL_TENSOR_PATTERNS: &[&str] = &[
    "attn_scale",
    "mlp_scale",
    "resid_mix",
    "q_gain",
    "skip_weight",
    "skip_weights",
];

pub struct TrainOutcome {
    pub train_losses: Vec<f32>,
    pub val_loss: Option<f32>,
    pub val_bpb: Option<f32>,
    pub quant_stats: QuantStats,
    pub raw_model_path: PathBuf,
    pub quantized_path: PathBuf,
    pub elapsed_ms: u128,
}

pub fn smoke_run() -> Result<TrainOutcome> {
    let cfg = Hyperparameters::smoke();
    run_with_config(cfg, None)
}

pub fn run_from_env() -> Result<TrainOutcome> {
    let cfg = Hyperparameters::from_env()?;
    run_with_config(cfg, Some(Device::Cpu))
}

pub fn run_with_config(cfg: Hyperparameters, force_device: Option<Device>) -> Result<TrainOutcome> {
    cfg.validate()?;
    let device = force_device.unwrap_or(Device::Cpu);
    let grad_accum_steps = cfg.grad_accum_steps(1)?;
    let model = Gpt::new(&cfg, &device)?;
    let mut optimizers = build_optimizers(&model, &cfg)?;

    if cfg.warmup_steps > 0 {
        let initial_state = snapshot_varmap(&model.varmap, &device)?;
        let mut warmup_loader = DistributedTokenLoader::new(&cfg.train_files_glob, 0, 1, &device)?;
        for warmup_step in 0..cfg.warmup_steps {
            let _ = train_step(
                &cfg,
                &model,
                &mut optimizers,
                &mut warmup_loader,
                grad_accum_steps,
            )?;
            if cfg.warmup_steps <= 20
                || (warmup_step + 1) % 10 == 0
                || warmup_step + 1 == cfg.warmup_steps
            {
                println!("warmup_step:{}/{}", warmup_step + 1, cfg.warmup_steps);
            }
        }
        restore_varmap(&model.varmap, &initial_state)?;
        optimizers = build_optimizers(&model, &cfg)?;
    }

    let mut loader = DistributedTokenLoader::new(&cfg.train_files_glob, 0, 1, &device)?;
    let val_tokens = load_validation_tokens(&cfg.val_files_glob, cfg.train_seq_len)?;
    let luts = build_sentencepiece_luts(&cfg.tokenizer_path, cfg.vocab_size)?;

    let start = Instant::now();
    let mut train_segment_start = Instant::now();
    let mut training_time_ms = 0.0_f64;
    let mut stop_after_step = None;
    let mut train_losses = Vec::new();
    let mut step = 0usize;
    let mut last_val_loss = None;
    let mut last_val_bpb = None;

    loop {
        let last_step = step == cfg.iterations || stop_after_step.is_some_and(|stop| step >= stop);
        let should_validate = last_step || (cfg.val_loss_every > 0 && step % cfg.val_loss_every == 0);
        if should_validate {
            training_time_ms += train_segment_start.elapsed().as_secs_f64() * 1000.0;
            let (val_loss, val_bpb) = evaluate(&cfg, &model, &device, grad_accum_steps, &val_tokens, &luts)?;
            last_val_loss = Some(val_loss);
            last_val_bpb = Some(val_bpb);
            println!(
                "step:{}/{} val_loss:{:.4} val_bpb:{:.4} train_time:{:.0}ms step_avg:{:.2}ms",
                step,
                cfg.iterations,
                val_loss,
                val_bpb,
                training_time_ms,
                training_time_ms / (step.max(1) as f64),
            );
            train_segment_start = Instant::now();
        }

        if last_step {
            if let Some(stop_step) = stop_after_step {
                if stop_step < cfg.iterations {
                    println!(
                        "stopping_early: wallclock_cap train_time:{:.0}ms step:{}/{}",
                        training_time_ms,
                        step,
                        cfg.iterations,
                    );
                }
            }
            break;
        }

        let elapsed_ms = training_time_ms + train_segment_start.elapsed().as_secs_f64() * 1000.0;
        let scale = lr_mul(&cfg, step, elapsed_ms);
        optimizers.set_learning_rates(&cfg, scale, step);
        let train_loss = train_step(
            &cfg,
            &model,
            &mut optimizers,
            &mut loader,
            grad_accum_steps,
        )?;
        train_losses.push(train_loss);
        step += 1;

        let approx_training_time_ms = training_time_ms + train_segment_start.elapsed().as_secs_f64() * 1000.0;
        if cfg.train_log_every > 0
            && (step <= 10 || step % cfg.train_log_every == 0 || stop_after_step.is_some())
        {
            println!(
                "step:{}/{} train_loss:{:.4} train_time:{:.0}ms step_avg:{:.2}ms",
                step,
                cfg.iterations,
                train_loss,
                approx_training_time_ms,
                approx_training_time_ms / step as f64,
            );
        }

        let reached_cap = cfg.max_wallclock_seconds > 0.0
            && approx_training_time_ms >= cfg.max_wallclock_seconds * 1000.0;
        if stop_after_step.is_none() && reached_cap {
            stop_after_step = Some(step);
        }
    }

    let output_dir = PathBuf::from("target/train-gpt-rs");
    std::fs::create_dir_all(&output_dir)?;
    let raw_model_path = output_dir.join("final_model.safetensors");
    model.varmap.save(&raw_model_path)?;
    let (quant_state, quant_stats) = quantize_varmap(&model.varmap)?;
    let quantized_path = output_dir.join("final_model.int8.json.z");
    save_quantized_state(&quant_state, &quantized_path)?;
    let roundtrip = load_quantized_state(&quantized_path)?;
    restore_quantized_state(&model.varmap, &roundtrip, &device)?;

    let (q_val_loss, q_val_bpb) = evaluate(&cfg, &model, &device, grad_accum_steps, &val_tokens, &luts)?;
    println!(
        "final_int8_zlib_roundtrip val_loss:{:.4} val_bpb:{:.4}",
        q_val_loss,
        q_val_bpb,
    );
    println!(
        "final_int8_zlib_roundtrip_exact val_loss:{:.8} val_bpb:{:.8}",
        q_val_loss,
        q_val_bpb,
    );

    Ok(TrainOutcome {
        train_losses,
        val_loss: last_val_loss,
        val_bpb: last_val_bpb,
        quant_stats,
        raw_model_path,
        quantized_path,
        elapsed_ms: start.elapsed().as_millis(),
    })
}

fn train_step(
    cfg: &Hyperparameters,
    model: &Gpt,
    optimizers: &mut Optimizers,
    loader: &mut DistributedTokenLoader,
    grad_accum_steps: usize,
) -> Result<f32> {
    let mut train_loss_sum = 0.0_f32;
    let mut total_loss: Option<Tensor> = None;
    for _ in 0..grad_accum_steps {
        let (x, y) = loader.next_batch(cfg.train_batch_tokens, cfg.train_seq_len, grad_accum_steps)?;
        let loss = model.forward_loss(&x, &y)?;
        train_loss_sum += loss.to_scalar::<f32>()?;
        let scaled_loss = loss.affine(1.0 / grad_accum_steps as f64, 0.0)?;
        total_loss = Some(match total_loss {
            Some(current) => current.broadcast_add(&scaled_loss)?,
            None => scaled_loss,
        });
    }
    let total_loss = total_loss.ok_or_else(|| anyhow!("missing accumulated loss"))?;
    let grads = total_loss.backward()?;
    if cfg.grad_clip_norm > 0.0 {
        bail!("GRAD_CLIP_NORM is not implemented in train-gpt-rs yet");
    }
    optimizers.step(&grads)?;
    Ok(train_loss_sum / grad_accum_steps as f32)
}

fn snapshot_varmap(varmap: &VarMap, device: &Device) -> Result<BTreeMap<String, Tensor>> {
    let lock = varmap.data().lock().unwrap();
    let mut state = BTreeMap::new();
    for (name, var) in lock.iter() {
        let shape = var.shape().dims().to_vec();
        let values = var
            .as_tensor()
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        state.insert(name.clone(), Tensor::from_vec(values, shape, device)?);
    }
    Ok(state)
}

fn restore_varmap(varmap: &VarMap, state: &BTreeMap<String, Tensor>) -> Result<()> {
    let mut lock = varmap.data().lock().unwrap();
    for (name, tensor) in state {
        let var = lock
            .get_mut(name)
            .ok_or_else(|| anyhow!("missing variable {name} during restore"))?;
        var.set(tensor)?;
    }
    Ok(())
}

fn lr_mul(cfg: &Hyperparameters, step: usize, elapsed_ms: f64) -> f64 {
    if cfg.warmdown_iters == 0 {
        return 1.0;
    }
    if cfg.max_wallclock_seconds <= 0.0 {
        let warmdown_start = cfg.iterations.saturating_sub(cfg.warmdown_iters);
        if (warmdown_start..cfg.iterations).contains(&step) {
            (cfg.iterations.saturating_sub(step)) as f64 / cfg.warmdown_iters.max(1) as f64
        } else {
            1.0
        }
    } else {
        let max_ms = cfg.max_wallclock_seconds * 1000.0;
        let step_ms = elapsed_ms / step.max(1) as f64;
        let warmdown_ms = cfg.warmdown_iters as f64 * step_ms;
        let remaining_ms = (max_ms - elapsed_ms).max(0.0);
        if remaining_ms <= warmdown_ms {
            remaining_ms / warmdown_ms.max(1e-9)
        } else {
            1.0
        }
    }
}

fn evaluate(
    cfg: &Hyperparameters,
    model: &Gpt,
    device: &Device,
    grad_accum_steps: usize,
    val_tokens: &[u32],
    luts: &TokenMetricLuts,
) -> Result<(f32, f32)> {
    let local_batch_tokens = cfg.val_batch_size / grad_accum_steps;
    if local_batch_tokens < cfg.train_seq_len {
        bail!("VAL_BATCH_SIZE must be at least TRAIN_SEQ_LEN after grad accumulation");
    }
    let local_batch_seqs = local_batch_tokens / cfg.train_seq_len;
    let total_seqs = (val_tokens.len() - 1) / cfg.train_seq_len;
    let mut val_loss_sum = 0.0_f64;
    let mut val_token_count = 0_usize;
    let mut val_byte_count = 0_usize;
    for batch_seq_start in (0..total_seqs).step_by(local_batch_seqs.max(1)) {
        let batch_seq_end = (batch_seq_start + local_batch_seqs).min(total_seqs);
        let raw_start = batch_seq_start * cfg.train_seq_len;
        let raw_end = batch_seq_end * cfg.train_seq_len + 1;
        let local = &val_tokens[raw_start..raw_end];
        let x = Tensor::from_vec(
            local[..local.len() - 1].to_vec(),
            ((local.len() - 1) / cfg.train_seq_len, cfg.train_seq_len),
            device,
        )?;
        let y = Tensor::from_vec(
            local[1..].to_vec(),
            ((local.len() - 1) / cfg.train_seq_len, cfg.train_seq_len),
            device,
        )?;
        let batch_loss = model.forward_loss(&x, &y)?.to_scalar::<f32>()?;
        val_loss_sum += f64::from(batch_loss) * y.elem_count() as f64;
        val_token_count += y.elem_count();
        let flat_x = x.flatten_all()?.to_vec1::<u32>()?;
        let flat_y = y.flatten_all()?.to_vec1::<u32>()?;
        for (prev, tgt) in flat_x.into_iter().zip(flat_y.into_iter()) {
            val_byte_count += luts.byte_count(prev, tgt)?;
        }
    }
    if val_token_count == 0 || val_byte_count == 0 {
        bail!("validation set did not produce any tokens or bytes");
    }
    let val_loss = (val_loss_sum / val_token_count as f64) as f32;
    let bits_per_token = f64::from(val_loss) / std::f64::consts::LN_2;
    let tokens_per_byte = val_token_count as f64 / val_byte_count as f64;
    Ok((val_loss, (bits_per_token * tokens_per_byte) as f32))
}

struct Optimizers {
    tok: AdamW,
    head: Option<AdamW>,
    scalar: AdamW,
    muon: Muon,
}

impl Optimizers {
    fn set_learning_rates(&mut self, cfg: &Hyperparameters, scale: f64, step: usize) {
        let frac = if cfg.muon_momentum_warmup_steps == 0 {
            1.0
        } else {
            (step as f64 / cfg.muon_momentum_warmup_steps as f64).min(1.0)
        };
        self.muon.momentum =
            (1.0 - frac) * cfg.muon_momentum_warmup_start + frac * cfg.muon_momentum;
        self.tok.set_learning_rate(
            if cfg.tie_embeddings {
                cfg.tied_embed_lr
            } else {
                cfg.embed_lr
            } * scale,
        );
        if let Some(head) = self.head.as_mut() {
            head.set_learning_rate(cfg.head_lr * scale);
        }
        self.scalar.set_learning_rate(cfg.scalar_lr * scale);
        self.muon.lr = cfg.matrix_lr * scale;
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        self.tok.step(grads)?;
        if let Some(head) = self.head.as_mut() {
            head.step(grads)?;
        }
        self.scalar.step(grads)?;
        self.muon.step(grads)
    }
}

fn build_optimizers(model: &Gpt, cfg: &Hyperparameters) -> Result<Optimizers> {
    let params = ParamsAdamW {
        lr: cfg.scalar_lr,
        beta1: cfg.beta1,
        beta2: cfg.beta2,
        eps: cfg.adam_eps,
        weight_decay: 0.0,
    };
    let mut tok = Vec::new();
    let mut head = Vec::new();
    let mut scalar = Vec::new();
    let mut matrix = Vec::new();
    let lock = model.varmap.data().lock().unwrap();
    let mut named = lock
        .iter()
        .map(|(name, var)| (name.clone(), var.clone()))
        .collect::<Vec<_>>();
    named.sort_by(|a, b| a.0.cmp(&b.0));
    drop(lock);
    for (name, var) in named {
        if name == "tok_emb.weight" {
            tok.push(var);
            continue;
        }
        if name == "lm_head.weight" {
            head.push(var);
            continue;
        }
        let dims = var.shape().dims();
        let is_control = CONTROL_TENSOR_PATTERNS
            .iter()
            .any(|pattern| name.contains(pattern));
        if dims.len() == 2 && !is_control {
            matrix.push((name, var));
        } else {
            scalar.push(var);
        }
    }

    Ok(Optimizers {
        tok: AdamW::new(
            tok,
            ParamsAdamW {
                lr: if cfg.tie_embeddings {
                    cfg.tied_embed_lr
                } else {
                    cfg.embed_lr
                },
                ..params.clone()
            },
        )?,
        head: if head.is_empty() {
            None
        } else {
            Some(AdamW::new(
                head,
                ParamsAdamW {
                    lr: cfg.head_lr,
                    ..params.clone()
                },
            )?)
        },
        scalar: AdamW::new(
            scalar,
            ParamsAdamW {
                lr: cfg.scalar_lr,
                ..params
            },
        )?,
        muon: Muon::new(
            matrix,
            cfg.matrix_lr,
            cfg.muon_momentum,
            cfg.muon_backend_steps,
        )?,
    })
}

struct Muon {
    vars: Vec<(String, Var)>,
    momentum_buffers: HashMap<String, Tensor>,
    lr: f64,
    momentum: f64,
    backend_steps: usize,
}

impl Muon {
    fn new(vars: Vec<(String, Var)>, lr: f64, momentum: f64, backend_steps: usize) -> Result<Self> {
        let mut momentum_buffers = HashMap::new();
        for (name, var) in &vars {
            momentum_buffers.insert(
                name.clone(),
                Tensor::zeros(var.shape(), candle_core::DType::F32, var.device())?,
            );
        }
        Ok(Self {
            vars,
            momentum_buffers,
            lr,
            momentum,
            backend_steps,
        })
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        for (name, var) in &self.vars {
            let Some(grad) = grads.get(var) else {
                continue;
            };
            let grad = grad.to_dtype(candle_core::DType::F32)?;
            let buf = self
                .momentum_buffers
                .get_mut(name)
                .expect("missing momentum buffer");
            let next_buf = (buf.affine(self.momentum, 0.0)? + &grad)?;
            let nesterov_grad = (&grad + &next_buf.affine(self.momentum, 0.0)?)?;
            *buf = next_buf;
            let mut update = zeropower_via_newton_schulz(&nesterov_grad, self.backend_steps)?;
            let (rows, cols) = update.dims2()?;
            let correction = if rows > cols {
                (rows as f64 / cols.max(1) as f64).sqrt()
            } else {
                1.0
            };
            update = update.affine(correction, 0.0)?;
            let next = (var.as_tensor() - update.affine(self.lr, 0.0)?)?;
            var.set(&next)?;
        }
        Ok(())
    }
}

fn zeropower_via_newton_schulz(grad: &Tensor, steps: usize) -> Result<Tensor> {
    let (rows, cols) = grad.dims2()?;
    let mut x = grad.to_dtype(DType::BF16)?;
    let norm = x.sqr()?.sum_all()?.sqrt()?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
    let scale = if norm == 0.0 { 1.0 } else { 1.0 / norm as f64 };
    x = x.affine(scale, 0.0)?;
    let transposed = rows > cols;
    if transposed {
        x = x.t()?;
    }
    for _ in 0..steps {
        let a = x.matmul(&x.t()?)?;
        let a2 = a.matmul(&a)?;
        let b = (a.affine(-4.7750, 0.0)? + a2.affine(2.0315, 0.0)?)?;
        x = (x.affine(3.4445, 0.0)? + b.matmul(&x)?)?;
    }
    let x = if transposed { x.t()? } else { x };
    Ok(x.to_dtype(DType::F32)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lr_mul_matches_python_step_warmdown() {
        let mut cfg = Hyperparameters::smoke();
        cfg.iterations = 20;
        cfg.warmdown_iters = 5;
        cfg.max_wallclock_seconds = 0.0;
        assert_eq!(lr_mul(&cfg, 0, 0.0), 1.0);
        assert_eq!(lr_mul(&cfg, 14, 0.0), 1.0);
        assert_eq!(lr_mul(&cfg, 15, 0.0), 1.0);
        assert!((lr_mul(&cfg, 16, 0.0) - 0.8).abs() < 1e-9);
        assert!((lr_mul(&cfg, 19, 0.0) - 0.2).abs() < 1e-9);
        assert_eq!(lr_mul(&cfg, 20, 0.0), 1.0);
    }

    #[test]
    fn grad_accumulation_matches_python_default() {
        let cfg = Hyperparameters::default();
        assert_eq!(cfg.grad_accum_steps(1).unwrap(), 8);
    }
}
