use anyhow::Result;
use train_gpt_rs::trainer::{run_from_env, smoke_run};

fn main() -> Result<()> {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "smoke".into());
    let outcome = match mode.as_str() {
        "smoke" => smoke_run()?,
        "train" => run_from_env()?,
        other => anyhow::bail!("unknown mode {other}; expected `smoke` or `train`"),
    };

    println!("elapsed_ms:{}", outcome.elapsed_ms);
    if let Some(loss) = outcome.val_loss {
        println!("val_loss:{loss:.4}");
    }
    if let Some(bpb) = outcome.val_bpb {
        println!("val_bpb:{bpb:.4}");
    }
    println!("raw_model:{}", outcome.raw_model_path.display());
    println!("quantized_model:{}", outcome.quantized_path.display());
    println!(
        "quant_stats:tensors:{} baseline_bytes:{} int8_bytes:{}",
        outcome.quant_stats.tensor_count,
        outcome.quant_stats.baseline_bytes,
        outcome.quant_stats.int8_bytes,
    );
    Ok(())
}
