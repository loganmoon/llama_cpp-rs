//! Grammar support for constraining the generation of tokens.
//! 
//! NOTE: The old grammar API has been deprecated in llama.cpp
//! This is a temporary stub to allow the codebase to build.
//! The new API uses sampler-based grammar integrated into the sampler chain.

use std::fmt::{Debug, Formatter};

/// A temporary stub for LlamaGrammar to maintain API compatibility
/// The actual implementation needs to be migrated to the new sampler-based grammar API
#[derive(Clone)]
pub struct LlamaGrammar {
    // Placeholder - the new API handles grammar through samplers
    _placeholder: (),
}

impl LlamaGrammar {
    /// Creates a new grammar - STUB IMPLEMENTATION
    pub fn new(_grammar: &str) -> Result<Self, String> {
        // This is a stub - actual implementation needs the new sampler API
        Ok(Self {
            _placeholder: (),
        })
    }
}

impl Debug for LlamaGrammar {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaGrammar")
            .field("stub", &"Grammar API needs migration to new sampler chain")
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