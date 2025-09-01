//! Sampler chain wrapper for the new llama.cpp sampler API
//!
//! This module provides a Rust-friendly interface to the new composable sampler chain
//! architecture in llama.cpp.

use llama_cpp_sys::{
    llama_context, llama_model, llama_model_get_vocab, llama_sampler, llama_sampler_accept,
    llama_sampler_chain_add, llama_sampler_chain_default_params, llama_sampler_chain_init,
    llama_sampler_chain_n, llama_sampler_chain_params, llama_sampler_free,
    llama_sampler_init_dist, llama_sampler_init_grammar, llama_sampler_init_greedy,
    llama_sampler_init_min_p, llama_sampler_init_mirostat, llama_sampler_init_mirostat_v2,
    llama_sampler_init_penalties, llama_sampler_init_temp, llama_sampler_init_temp_ext,
    llama_sampler_init_top_k, llama_sampler_init_top_p, llama_sampler_init_typical,
    llama_sampler_reset, llama_sampler_sample, llama_token, llama_vocab,
};
use std::ptr::NonNull;

use crate::Token;

/// A chain of samplers that process tokens sequentially
pub struct SamplerChain {
    chain: NonNull<llama_sampler>,
    /// Keep track of individual samplers to prevent use-after-free
    _samplers: Vec<SamplerHandle>,
}

/// Handle to an individual sampler
struct SamplerHandle(NonNull<llama_sampler>);

/// Types of samplers that can be added to the chain
#[derive(Debug, Clone)]
pub enum SamplerType {
    /// Greedy sampling - always select the most probable token
    Greedy,
    /// Distribution sampling with a random seed
    Dist { seed: u32 },
    /// Temperature sampling
    Temperature { t: f32 },
    /// Extended temperature with delta and exponent
    TempExt { t: f32, delta: f32, exponent: f32 },
    /// Top-K sampling
    TopK { k: i32 },
    /// Top-P (nucleus) sampling
    TopP { p: f32, min_keep: usize },
    /// Min-P sampling
    MinP { p: f32, min_keep: usize },
    /// Typical sampling
    Typical { p: f32, min_keep: usize },
    /// Repetition penalties
    Penalties {
        tokens: Vec<i32>,
        penalty_last_n: i32,
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_present: f32,
    },
    /// Grammar-constrained sampling
    Grammar {
        vocab: *const llama_vocab,
        grammar_str: String,
        grammar_root: Option<String>,
    },
    /// Mirostat sampling v1
    Mirostat { tau: f32, eta: f32, m: i32 },
    /// Mirostat sampling v2
    MirostatV2 { tau: f32, eta: f32 },
}

/// Error types for sampler chain operations
#[derive(Debug)]
pub enum SamplerChainError {
    /// Null pointer returned from FFI
    NullPointer,
    /// Invalid parameter value
    InvalidParameter(String),
}

impl std::fmt::Display for SamplerChainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SamplerChainError::NullPointer => write!(f, "Null pointer returned from llama.cpp"),
            SamplerChainError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
        }
    }
}

impl std::error::Error for SamplerChainError {}

impl SamplerChain {
    /// Create a new empty sampler chain
    pub fn new() -> Result<Self, SamplerChainError> {
        let params = unsafe { llama_sampler_chain_default_params() };
        let chain = unsafe { llama_sampler_chain_init(params) };
        
        NonNull::new(chain)
            .map(|chain| Self {
                chain,
                _samplers: Vec::new(),
            })
            .ok_or(SamplerChainError::NullPointer)
    }
    
    /// Add a sampler to the chain
    pub fn add(&mut self, sampler_type: SamplerType) -> Result<(), SamplerChainError> {
        let sampler = match sampler_type {
            SamplerType::Greedy => unsafe { llama_sampler_init_greedy() },
            SamplerType::Dist { seed } => unsafe { llama_sampler_init_dist(seed) },
            SamplerType::Temperature { t } => unsafe { llama_sampler_init_temp(t) },
            SamplerType::TempExt { t, delta, exponent } => {
                unsafe { llama_sampler_init_temp_ext(t, delta, exponent) }
            }
            SamplerType::TopK { k } => unsafe { llama_sampler_init_top_k(k) },
            SamplerType::TopP { p, min_keep } => {
                unsafe { llama_sampler_init_top_p(p, min_keep) }
            }
            SamplerType::MinP { p, min_keep } => {
                unsafe { llama_sampler_init_min_p(p, min_keep) }
            }
            SamplerType::Typical { p, min_keep } => {
                unsafe { llama_sampler_init_typical(p, min_keep) }
            }
            SamplerType::Penalties {
                tokens: _,
                penalty_last_n,
                penalty_repeat,
                penalty_freq,
                penalty_present,
            } => {
                // The new API uses llama_sampler_init_penalties with 4 parameters
                unsafe {
                    llama_sampler_init_penalties(
                        penalty_last_n,
                        penalty_repeat,
                        penalty_freq,
                        penalty_present,
                    )
                }
            }
            SamplerType::Grammar { vocab, grammar_str, grammar_root } => {
                use std::ffi::CString;
                let grammar_cstr = CString::new(grammar_str)
                    .map_err(|_| SamplerChainError::InvalidParameter("Invalid grammar string".to_string()))?;
                let root_cstr = match grammar_root {
                    Some(root) => {
                        let cstr = CString::new(root)
                            .map_err(|_| SamplerChainError::InvalidParameter("Invalid grammar root".to_string()))?;
                        Some(cstr)
                    }
                    None => None,
                };
                
                // Grammar creation needs to be handled specially
                // since we need to return from the outer function on error
                let sampler = unsafe {
                    llama_sampler_init_grammar(
                        vocab,
                        grammar_cstr.as_ptr(),
                        root_cstr.as_ref().map(|s| s.as_ptr()).unwrap_or(std::ptr::null()),
                    )
                };
                sampler
            }
            SamplerType::Mirostat { tau, eta, m } => {
                // mirostat v1 needs n_vocab and seed parameters
                // For now, we'll use reasonable defaults
                unsafe { llama_sampler_init_mirostat(32000, 42, tau, eta, m) }
            }
            SamplerType::MirostatV2 { tau, eta } => {
                // mirostat v2 needs a seed parameter
                unsafe { llama_sampler_init_mirostat_v2(42, tau, eta) }
            }
        };
        
        NonNull::new(sampler)
            .map(|sampler_nn| {
                unsafe {
                    llama_sampler_chain_add(self.chain.as_ptr(), sampler_nn.as_ptr());
                }
                self._samplers.push(SamplerHandle(sampler_nn));
            })
            .ok_or(SamplerChainError::NullPointer)
    }
    
    /// Sample a token from the chain
    pub fn sample(&mut self, ctx: *mut llama_context, idx: i32) -> Token {
        let token_id = unsafe {
            llama_sampler_sample(self.chain.as_ptr(), ctx, idx)
        };
        Token(token_id)
    }
    
    /// Accept a token for context updates
    pub fn accept(&mut self, token: Token) {
        unsafe {
            llama_sampler_accept(self.chain.as_ptr(), token.0);
        }
    }
    
    /// Reset the sampler chain state
    pub fn reset(&mut self) {
        unsafe {
            llama_sampler_reset(self.chain.as_ptr());
        }
    }
    
    /// Get the number of samplers in the chain
    pub fn len(&self) -> usize {
        unsafe { llama_sampler_chain_n(self.chain.as_ptr()) as usize }
    }
    
    /// Check if the chain is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Drop for SamplerChain {
    fn drop(&mut self) {
        unsafe {
            llama_sampler_free(self.chain.as_ptr());
        }
    }
}

impl Drop for SamplerHandle {
    fn drop(&mut self) {
        // Individual samplers are freed when the chain is freed,
        // so we don't need to free them here
    }
}

/// Builder for creating a sampler chain with a fluent API
pub struct SamplerChainBuilder {
    chain: SamplerChain,
}

impl SamplerChainBuilder {
    /// Create a new sampler chain builder
    pub fn new() -> Result<Self, SamplerChainError> {
        Ok(Self {
            chain: SamplerChain::new()?,
        })
    }
    
    /// Add a greedy sampler
    pub fn greedy(mut self) -> Result<Self, SamplerChainError> {
        self.chain.add(SamplerType::Greedy)?;
        Ok(self)
    }
    
    /// Add a temperature sampler
    pub fn temperature(mut self, t: f32) -> Result<Self, SamplerChainError> {
        self.chain.add(SamplerType::Temperature { t })?;
        Ok(self)
    }
    
    /// Add a top-k sampler
    pub fn top_k(mut self, k: i32) -> Result<Self, SamplerChainError> {
        self.chain.add(SamplerType::TopK { k })?;
        Ok(self)
    }
    
    /// Add a top-p sampler
    pub fn top_p(mut self, p: f32, min_keep: usize) -> Result<Self, SamplerChainError> {
        self.chain.add(SamplerType::TopP { p, min_keep })?;
        Ok(self)
    }
    
    /// Add a min-p sampler
    pub fn min_p(mut self, p: f32, min_keep: usize) -> Result<Self, SamplerChainError> {
        self.chain.add(SamplerType::MinP { p, min_keep })?;
        Ok(self)
    }
    
    /// Add a typical sampler
    pub fn typical(mut self, p: f32, min_keep: usize) -> Result<Self, SamplerChainError> {
        self.chain.add(SamplerType::Typical { p, min_keep })?;
        Ok(self)
    }
    
    /// Add repetition penalties
    pub fn penalties(
        mut self,
        penalty_last_n: i32,
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_present: f32,
    ) -> Result<Self, SamplerChainError> {
        self.chain.add(SamplerType::Penalties {
            tokens: vec![],
            penalty_last_n,
            penalty_repeat,
            penalty_freq,
            penalty_present,
        })?;
        Ok(self)
    }
    
    /// Add a distribution sampler with seed
    pub fn dist(mut self, seed: u32) -> Result<Self, SamplerChainError> {
        self.chain.add(SamplerType::Dist { seed })?;
        Ok(self)
    }
    
    /// Build the final sampler chain
    pub fn build(self) -> SamplerChain {
        self.chain
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sampler_chain_creation() {
        let chain = SamplerChain::new();
        assert!(chain.is_ok());
        let chain = chain.unwrap();
        assert!(chain.is_empty());
    }
    
    #[test]
    fn test_sampler_chain_builder() {
        let result = SamplerChainBuilder::new()
            .and_then(|b| b.top_k(50))
            .and_then(|b| b.temperature(0.8))
            .and_then(|b| b.dist(42));
        
        assert!(result.is_ok());
        let chain = result.unwrap().build();
        assert_eq!(chain.len(), 3);
    }
}