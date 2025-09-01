//! Grammar support for constraining the generation of tokens.
//! 
//! This module provides grammar support through the new sampler-based API.
//! Grammar constraints are now integrated into the sampler chain rather than
//! being a separate system.

use llama_cpp_sys::{llama_model_get_vocab, llama_vocab};
use std::fmt::{Debug, Formatter};
use std::ptr::NonNull;

/// A grammar definition that can be used to constrain token generation
/// 
/// The grammar is stored as a string and optional root rule, which will be
/// used to create a grammar sampler when needed.
#[derive(Clone)]
pub struct LlamaGrammar {
    /// The grammar string (e.g., BNF format)
    grammar_str: String,
    /// Optional root rule for the grammar
    grammar_root: Option<String>,
}

impl LlamaGrammar {
    /// Creates a new grammar from a grammar string
    pub fn new(grammar: &str) -> Result<Self, String> {
        Ok(Self {
            grammar_str: grammar.to_string(),
            grammar_root: None,
        })
    }
    
    /// Creates a new grammar with a specified root rule
    pub fn with_root(grammar: &str, root: &str) -> Result<Self, String> {
        Ok(Self {
            grammar_str: grammar.to_string(),
            grammar_root: Some(root.to_string()),
        })
    }
    
    /// Get the grammar string
    pub fn grammar_str(&self) -> &str {
        &self.grammar_str
    }
    
    /// Get the optional root rule
    pub fn grammar_root(&self) -> Option<&str> {
        self.grammar_root.as_deref()
    }
    
    /// Convert this grammar to a sampler configuration
    /// Requires a model pointer to get the vocabulary
    pub fn to_sampler_type(&self, model: *const llama_cpp_sys::llama_model) -> crate::sampler_chain::SamplerType {
        let vocab = unsafe { llama_model_get_vocab(model) };
        crate::sampler_chain::SamplerType::Grammar {
            vocab,
            grammar_str: self.grammar_str.clone(),
            grammar_root: self.grammar_root.clone(),
        }
    }
}

impl Debug for LlamaGrammar {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaGrammar")
            .field("grammar_str", &self.grammar_str)
            .field("grammar_root", &self.grammar_root)
            .finish()
    }
}

unsafe impl Send for LlamaGrammar {}
unsafe impl Sync for LlamaGrammar {}

/// Grammar stage placeholder
#[derive(Clone, Debug)]
pub struct GrammarStage {
    pub(crate) grammar: LlamaGrammar,
    pub(crate) accepted_up_to: Option<usize>,
}