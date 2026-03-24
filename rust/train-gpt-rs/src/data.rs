use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail, ensure};
use candle_core::{Device, Tensor};

const SHARD_MAGIC: i32 = 20240520;
const SHARD_VERSION: i32 = 1;
const HEADER_INTS: usize = 256;
const HEADER_BYTES: usize = HEADER_INTS * 4;

pub fn resolve_glob(pattern: &str) -> Result<Vec<PathBuf>> {
    if let Some(star) = pattern.find('*') {
        let slash = pattern[..star].rfind('/');
        let (dir, prefix) = match slash {
            Some(idx) => (&pattern[..idx], &pattern[idx + 1..star]),
            None => (".", &pattern[..star]),
        };
        let dir = Path::new(dir);
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            if name.starts_with(prefix) {
                out.push(path);
            }
        }
        out.sort();
        return Ok(out);
    }

    let path = PathBuf::from(pattern);
    if path.exists() {
        Ok(vec![path])
    } else {
        Ok(Vec::new())
    }
}

pub fn load_data_shard(path: &Path) -> Result<Vec<u16>> {
    let bytes = fs::read(path)?;
    ensure!(
        bytes.len() >= HEADER_BYTES,
        "{} is too small to contain a shard header",
        path.display()
    );

    let mut header = [0_i32; HEADER_INTS];
    for (index, slot) in header.iter_mut().enumerate() {
        let offset = index * 4;
        *slot = i32::from_le_bytes(bytes[offset..offset + 4].try_into()?);
    }

    ensure!(
        header[0] == SHARD_MAGIC,
        "unexpected shard magic for {}",
        path.display()
    );
    ensure!(
        header[1] == SHARD_VERSION,
        "unexpected shard version for {}",
        path.display()
    );

    let num_tokens = usize::try_from(header[2])
        .map_err(|_| anyhow::anyhow!("negative token count in {}", path.display()))?;
    let expected_size = HEADER_BYTES + num_tokens * 2;
    ensure!(
        bytes.len() == expected_size,
        "shard size mismatch for {}",
        path.display()
    );

    let mut tokens = Vec::with_capacity(num_tokens);
    for chunk in bytes[HEADER_BYTES..].chunks_exact(2) {
        tokens.push(u16::from_le_bytes(chunk.try_into()?));
    }
    ensure!(
        tokens.len() == num_tokens,
        "short token payload in {}",
        path.display()
    );
    Ok(tokens)
}

pub fn load_validation_tokens(pattern: &str, seq_len: usize) -> Result<Vec<u32>> {
    let files = resolve_glob(pattern)?;
    if files.is_empty() {
        bail!("no files found for pattern {pattern}");
    }
    let mut tokens = Vec::new();
    for file in files {
        tokens.extend(load_data_shard(&file)?.into_iter().map(u32::from));
    }
    if tokens.len() <= seq_len {
        bail!("validation split is too short for TRAIN_SEQ_LEN={seq_len}");
    }
    let usable = ((tokens.len() - 1) / seq_len) * seq_len;
    tokens.truncate(usable + 1);
    Ok(tokens)
}

#[derive(Debug, Clone)]
pub struct TokenStream {
    files: Vec<PathBuf>,
    file_idx: usize,
    tokens: Vec<u16>,
    pos: usize,
}

impl TokenStream {
    pub fn new(pattern: &str) -> Result<Self> {
        let files = resolve_glob(pattern)?;
        if files.is_empty() {
            bail!("no files found for pattern {pattern}");
        }
        let tokens = load_data_shard(&files[0])?;
        Ok(Self {
            files,
            file_idx: 0,
            tokens,
            pos: 0,
        })
    }

    fn advance_file(&mut self) -> Result<()> {
        self.file_idx = (self.file_idx + 1) % self.files.len();
        self.tokens = load_data_shard(&self.files[self.file_idx])?;
        self.pos = 0;
        Ok(())
    }

    pub fn take(&mut self, n: usize) -> Result<Vec<u16>> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let avail = self.tokens.len().saturating_sub(self.pos);
            if avail == 0 {
                self.advance_file()?;
                continue;
            }
            let k = (n - out.len()).min(avail);
            out.extend_from_slice(&self.tokens[self.pos..self.pos + k]);
            self.pos += k;
        }
        Ok(out)
    }
}

pub struct DistributedTokenLoader {
    rank: usize,
    world_size: usize,
    device: Device,
    stream: TokenStream,
}

impl DistributedTokenLoader {
    pub fn new(pattern: &str, rank: usize, world_size: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            rank,
            world_size,
            device: device.clone(),
            stream: TokenStream::new(pattern)?,
        })
    }

    pub fn next_batch(
        &mut self,
        global_tokens: usize,
        seq_len: usize,
        grad_accum_steps: usize,
    ) -> Result<(Tensor, Tensor)> {
        let local_tokens = global_tokens / (self.world_size * grad_accum_steps);
        let per_rank_span = local_tokens + 1;
        let chunk = self.stream.take(per_rank_span * self.world_size)?;
        let start = self.rank * per_rank_span;
        let local = &chunk[start..start + per_rank_span];
        let x: Vec<u32> = local[..local.len() - 1]
            .iter()
            .copied()
            .map(u32::from)
            .collect();
        let y: Vec<u32> = local[1..].iter().copied().map(u32::from).collect();
        let batch = x.len() / seq_len;
        Ok((
            Tensor::from_vec(x, (batch, seq_len), &self.device)?,
            Tensor::from_vec(y, (batch, seq_len), &self.device)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn write_shard(path: &Path, tokens: &[u16]) -> Result<()> {
        let mut bytes = vec![0_u8; HEADER_BYTES];
        let mut ints = [0_i32; HEADER_INTS];
        ints[0] = SHARD_MAGIC;
        ints[1] = SHARD_VERSION;
        ints[2] = i32::try_from(tokens.len())?;
        for (index, value) in ints.into_iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        for token in tokens {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        let mut file = fs::File::create(path)?;
        file.write_all(&bytes)?;
        Ok(())
    }

    #[test]
    fn shard_roundtrip_and_loader_wraparound() -> Result<()> {
        let dir = tempdir()?;
        write_shard(&dir.path().join("fineweb_train_000000.bin"), &[1, 2, 3])?;
        write_shard(&dir.path().join("fineweb_train_000001.bin"), &[4, 5])?;
        let mut stream =
            TokenStream::new(&format!("{}/fineweb_train_*.bin", dir.path().display()))?;
        assert_eq!(stream.take(6)?, vec![1, 2, 3, 4, 5, 1]);
        Ok(())
    }
}
