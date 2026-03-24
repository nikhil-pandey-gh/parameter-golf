use std::path::Path;

use anyhow::{Result, bail, ensure};
use sentencepiece_model::{SentencePieceModel, Type};

#[derive(Debug, Clone)]
pub struct TokenMetricLuts {
    pub base_bytes: Vec<i16>,
    pub has_leading_space: Vec<bool>,
    pub is_boundary_token: Vec<bool>,
}

impl TokenMetricLuts {
    pub fn byte_count(&self, prev_id: u32, tgt_id: u32) -> Result<usize> {
        let prev = usize::try_from(prev_id)?;
        let tgt = usize::try_from(tgt_id)?;
        ensure!(
            prev < self.is_boundary_token.len(),
            "prev token id {prev_id} out of range"
        );
        ensure!(
            tgt < self.base_bytes.len(),
            "target token id {tgt_id} out of range"
        );
        let mut bytes = i32::from(self.base_bytes[tgt]);
        if self.has_leading_space[tgt] && !self.is_boundary_token[prev] {
            bytes += 1;
        }
        usize::try_from(bytes.max(0)).map_err(Into::into)
    }
}

pub fn build_sentencepiece_luts(path: &Path, vocab_size: usize) -> Result<TokenMetricLuts> {
    let model = SentencePieceModel::from_file(path)?;
    let pieces = model.pieces();
    ensure!(
        pieces.len() == vocab_size,
        "VOCAB_SIZE={vocab_size} does not match tokenizer vocab_size={}",
        pieces.len()
    );

    let mut base_bytes = vec![0_i16; vocab_size];
    let mut has_leading_space = vec![false; vocab_size];
    let mut is_boundary_token = vec![true; vocab_size];

    for (token_id, piece) in pieces.iter().enumerate() {
        let piece_type =
            Type::try_from(piece.r#type.unwrap_or(Type::Normal as i32)).unwrap_or(Type::Normal);
        match piece_type {
            Type::Control | Type::Unknown | Type::Unused => continue,
            Type::Byte => {
                base_bytes[token_id] = 1;
                is_boundary_token[token_id] = false;
            }
            Type::Normal | Type::UserDefined => {
                is_boundary_token[token_id] = false;
                let mut value = piece.piece.clone().unwrap_or_default();
                if let Some(stripped) = value.strip_prefix('▁') {
                    has_leading_space[token_id] = true;
                    value = stripped.to_owned();
                }
                base_bytes[token_id] = i16::try_from(value.as_bytes().len())?;
            }
        }
    }

    Ok(TokenMetricLuts {
        base_bytes,
        has_leading_space,
        is_boundary_token,
    })
}

pub fn maybe_build_sentencepiece_luts(
    path: &Path,
    vocab_size: usize,
) -> Result<Option<TokenMetricLuts>> {
    if !path.exists() {
        return Ok(None);
    }
    match build_sentencepiece_luts(path, vocab_size) {
        Ok(luts) => Ok(Some(luts)),
        Err(err) => {
            if err.to_string().contains("does not match tokenizer") {
                bail!(err)
            }
            Ok(None)
        }
    }
}
