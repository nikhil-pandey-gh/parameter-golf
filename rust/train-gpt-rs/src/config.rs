use std::env;
use std::path::PathBuf;

use anyhow::{Result, bail};

#[derive(Debug, Clone)]
pub struct Hyperparameters {
    pub data_path: PathBuf,
    pub train_files_glob: String,
    pub val_files_glob: String,
    pub tokenizer_path: PathBuf,
    pub run_id: String,
    pub seed: u64,
    pub val_batch_size: usize,
    pub val_loss_every: usize,
    pub train_log_every: usize,
    pub iterations: usize,
    pub warmdown_iters: usize,
    pub warmup_steps: usize,
    pub train_batch_tokens: usize,
    pub train_seq_len: usize,
    pub max_wallclock_seconds: f64,
    pub qk_gain_init: f64,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub model_dim: usize,
    pub num_heads: usize,
    pub mlp_mult: usize,
    pub tie_embeddings: bool,
    pub rope_base: f64,
    pub logit_softcap: f64,
    pub embed_lr: f64,
    pub head_lr: f64,
    pub tied_embed_lr: f64,
    pub tied_embed_init_std: f64,
    pub matrix_lr: f64,
    pub scalar_lr: f64,
    pub muon_momentum: f64,
    pub muon_backend_steps: usize,
    pub muon_momentum_warmup_start: f64,
    pub muon_momentum_warmup_steps: usize,
    pub beta1: f64,
    pub beta2: f64,
    pub adam_eps: f64,
    pub grad_clip_norm: f64,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        let data_path = PathBuf::from(
            env::var("DATA_PATH").unwrap_or_else(|_| "./data/datasets/fineweb10B_sp1024".into()),
        );
        let tokenizer_path = PathBuf::from(
            env::var("TOKENIZER_PATH")
                .unwrap_or_else(|_| "./data/tokenizers/fineweb_1024_bpe.model".into()),
        );
        Self {
            train_files_glob: data_path.join("fineweb_train_*.bin").display().to_string(),
            val_files_glob: data_path.join("fineweb_val_*.bin").display().to_string(),
            data_path,
            tokenizer_path,
            run_id: env::var("RUN_ID").unwrap_or_else(|_| "rust_smoke".into()),
            seed: env_parse("SEED", 1337_u64),
            val_batch_size: env_parse("VAL_BATCH_SIZE", 524_288_usize),
            val_loss_every: env_parse("VAL_LOSS_EVERY", 1_000_usize),
            train_log_every: env_parse("TRAIN_LOG_EVERY", 200_usize),
            iterations: env_parse("ITERATIONS", 20_000_usize),
            warmdown_iters: env_parse("WARMDOWN_ITERS", 1_200_usize),
            warmup_steps: env_parse("WARMUP_STEPS", 20_usize),
            train_batch_tokens: env_parse("TRAIN_BATCH_TOKENS", 524_288_usize),
            train_seq_len: env_parse("TRAIN_SEQ_LEN", 1_024_usize),
            max_wallclock_seconds: env_parse("MAX_WALLCLOCK_SECONDS", 600.0_f64),
            qk_gain_init: env_parse("QK_GAIN_INIT", 1.5_f64),
            vocab_size: env_parse("VOCAB_SIZE", 1_024_usize),
            num_layers: env_parse("NUM_LAYERS", 9_usize),
            num_kv_heads: env_parse("NUM_KV_HEADS", 4_usize),
            model_dim: env_parse("MODEL_DIM", 512_usize),
            num_heads: env_parse("NUM_HEADS", 8_usize),
            mlp_mult: env_parse("MLP_MULT", 2_usize),
            tie_embeddings: env_parse::<u8>("TIE_EMBEDDINGS", 1) != 0,
            rope_base: env_parse("ROPE_BASE", 10_000.0_f64),
            logit_softcap: env_parse("LOGIT_SOFTCAP", 30.0_f64),
            embed_lr: env_parse("EMBED_LR", 0.6_f64),
            head_lr: env_parse("HEAD_LR", 0.008_f64),
            tied_embed_lr: env_parse("TIED_EMBED_LR", 0.05_f64),
            tied_embed_init_std: env_parse("TIED_EMBED_INIT_STD", 0.005_f64),
            matrix_lr: env_parse("MATRIX_LR", 0.04_f64),
            scalar_lr: env_parse("SCALAR_LR", 0.04_f64),
            muon_momentum: env_parse("MUON_MOMENTUM", 0.95_f64),
            muon_backend_steps: env_parse("MUON_BACKEND_STEPS", 5_usize),
            muon_momentum_warmup_start: env_parse("MUON_MOMENTUM_WARMUP_START", 0.85_f64),
            muon_momentum_warmup_steps: env_parse("MUON_MOMENTUM_WARMUP_STEPS", 500_usize),
            beta1: env_parse("BETA1", 0.9_f64),
            beta2: env_parse("BETA2", 0.95_f64),
            adam_eps: env_parse("ADAM_EPS", 1e-8_f64),
            grad_clip_norm: env_parse("GRAD_CLIP_NORM", 0.0_f64),
        }
    }
}

impl Hyperparameters {
    pub fn from_env() -> Result<Self> {
        let cfg = Self::default();
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn smoke() -> Self {
        let mut cfg = Self::default();
        cfg.run_id = "rust_smoke".into();
        cfg.vocab_size = 128;
        cfg.num_layers = 4;
        cfg.num_heads = 4;
        cfg.num_kv_heads = 2;
        cfg.model_dim = 64;
        cfg.mlp_mult = 2;
        cfg.train_seq_len = 16;
        cfg.train_batch_tokens = 128;
        cfg.val_batch_size = 128;
        cfg.iterations = 2;
        cfg.warmup_steps = 0;
        cfg.warmdown_iters = 0;
        cfg.val_loss_every = 1;
        cfg.train_log_every = 1;
        cfg
    }

    pub fn grad_accum_steps(&self, world_size: usize) -> Result<usize> {
        if world_size == 0 {
            bail!("world_size must be positive");
        }
        if 8 % world_size != 0 {
            bail!("world_size={world_size} must divide 8 to mirror train_gpt.py");
        }
        Ok(8 / world_size)
    }

    pub fn local_batch_tokens(&self, world_size: usize, grad_accum_steps: usize) -> Result<usize> {
        let denom = world_size
            .checked_mul(grad_accum_steps)
            .ok_or_else(|| anyhow::anyhow!("local batch denominator overflow"))?;
        let local = self.train_batch_tokens / denom;
        if local < self.train_seq_len {
            bail!(
                "TRAIN_BATCH_TOKENS={} is too small for TRAIN_SEQ_LEN={} with world_size={} grad_accum_steps={}",
                self.train_batch_tokens,
                self.train_seq_len,
                world_size,
                grad_accum_steps
            );
        }
        Ok(local)
    }

    pub fn validate(&self) -> Result<()> {
        if self.logit_softcap <= 0.0 {
            bail!("LOGIT_SOFTCAP must be positive");
        }
        if self.num_heads == 0 || self.num_kv_heads == 0 {
            bail!("NUM_HEADS and NUM_KV_HEADS must be positive");
        }
        if self.model_dim % self.num_heads != 0 {
            bail!("MODEL_DIM must be divisible by NUM_HEADS");
        }
        if self.num_heads % self.num_kv_heads != 0 {
            bail!("NUM_HEADS must be divisible by NUM_KV_HEADS");
        }
        if (self.model_dim / self.num_heads) % 2 != 0 {
            bail!("head_dim must be even for RoPE");
        }
        if self.train_seq_len == 0 || self.train_batch_tokens == 0 {
            bail!("TRAIN_SEQ_LEN and TRAIN_BATCH_TOKENS must be positive");
        }
        if self.vocab_size == 0 || self.model_dim == 0 || self.num_layers == 0 {
            bail!("VOCAB_SIZE, MODEL_DIM, and NUM_LAYERS must be positive");
        }
        Ok(())
    }
}

fn env_parse<T>(key: &str, default: T) -> T
where
    T: std::str::FromStr,
{
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<T>().ok())
        .unwrap_or(default)
}
