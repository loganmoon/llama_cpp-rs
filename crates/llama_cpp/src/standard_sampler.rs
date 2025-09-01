use std::ptr::addr_of_mut;

use llama_cpp_sys::{
    llama_context, llama_get_model, llama_token, llama_token_data_array,
};

// The following sampling functions are deprecated in the new API:
// llama_grammar_accept_token, llama_sample_entropy, llama_sample_grammar,
// llama_sample_min_p, llama_sample_repetition_penalties, llama_sample_tail_free,
// llama_sample_temp, llama_sample_token, llama_sample_token_greedy, llama_sample_token_mirostat,
// llama_sample_token_mirostat_v2, llama_sample_top_k, llama_sample_top_p, llama_sample_typical
// Use the new sampler chain API instead (feature = "use_new_sampler_api")

use crate::{grammar::LlamaGrammar, Sampler, Token};

use crate::sampler_chain::{SamplerChain, SamplerChainError, SamplerType};

/// Functions which modify the probability distribution output by the model.
///
/// Standard ordering for samplers (taken from [kobold.cpp](https://github.com/LostRuins/koboldcpp)):
///
/// 1. [`SamplerStage::Grammar`]
/// 2. [`SamplerStage::RepetitionPenalty`]
/// 3. [`SamplerStage::Temperature`], [SamplerStage::DynamicTemperature]
/// 4. [`SamplerStage::TopK`]
/// 5. [`SamplerStage::TailFree`]
/// 6. [`SamplerStage::Typical`]
/// 7. [`SamplerStage::TopP`], [`SamplerStage::MinP`]
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum SamplerStage {
    /// Divide the logits by this value. Ranges from 0 to 2. Lower values yield a more
    /// deterministic output, and higher values yield a more random/creative output. This should
    /// not be used with [`SamplerStage::DynamicTemperature`].
    Temperature(f32),

    /// Divide the logits by a dynamically determined value between `min_temp`
    /// and `max_temp`. This should not be used with [`SamplerStage::Temperature`].
    ///
    /// This determines the temperature using the equation:
    ///
    /// ```
    /// (current_entropy / maximum_entropy) ^ exponent_val
    /// ```
    ///
    /// where `current_entropy` is the entropy of the current
    /// distribution over tokens, `maximum_entropy` is the maximum possible
    /// entropy over that distribution, and `exponent_val` is the parameter below.
    ///
    /// See: <https://arxiv.org/pdf/2309.02772.pdf>
    DynamicTemperature {
        /// Determines the minimum possible temperature for this stage. Should be between 0 and 2.
        min_temp: f32,

        /// Determines the maximum possible temperature for this stage. Should be between 0 and 2.
        max_temp: f32,

        /// The `exponent_val` parameter. 1 is a good starting point. Values less than 1 cause the
        /// temperature to approach `max_temp` more quickly at small entropies.
        exponent_val: f32,
    },
    /// Penalizes generating a token that is within the `last_n` tokens of context in various ways.
    RepetitionPenalty {
        /// Divide the token's logit by this value if they appear one or more time in the `last_n`
        /// tokens. 1.0 disables this, and values from 1.0-1.2 work well.
        ///
        /// See page 5 of <https://arxiv.org/pdf/1909.05858.pdf>
        repetition_penalty: f32,

        /// Subtract this value from the token's logit for each time the token appears in the
        /// `last_n` tokens. 0.0 disables this, and 0.0-1.0 are reasonable values.
        ///
        /// See: <https://platform.openai.com/docs/guides/text-generation/parameter-details>
        frequency_penalty: f32,

        /// Subtract this value from the token's logit if the token appears in the `last_n` tokens.
        /// 0.0 disables this, and 0.0-1.0 are reasonable values.
        ///
        /// See: <https://platform.openai.com/docs/guides/text-generation/parameter-details>
        presence_penalty: f32,

        /// How many tokens back to look when determining penalties. -1 means context size, and 0
        /// disables this stage.
        last_n: i32,
    },

    /// Keep the most likely tokens until their total probability exceeds `p`.
    ///
    /// See: <https://arxiv.org/abs/1904.09751>
    TopP(f32),

    /// Remove tokens with probability less than `p` times the probability of the most likely
    /// token.
    ///
    /// See: <https://github.com/ggerganov/llama.cpp/pull/3841>
    MinP(f32),

    /// Keep the `k` tokens with the highest probability.
    ///
    /// See: <https://arxiv.org/abs/1904.09751>
    TopK(i32),

    /// Typical Sampling
    ///
    /// See: <https://arxiv.org/abs/2202.00666>
    Typical(f32),

    /// Tail Free Sampling
    ///
    /// See: <https://www.trentonbricken.com/Tail-Free-Sampling/>
    TailFree(f32),

    /// A stage that uses a [`LlamaGrammar`] to remove tokens that do not align with a given
    /// grammar. Since this stage has to handle mutable state, an instance of this stage should
    /// only be used in one completion.
    ///
    /// See [`GrammarStage`] and [`LlamaGrammar`] for more information.
    Grammar(GrammarStage),
}

impl SamplerStage {
    /// Creates a new [`SamplerStage::Grammar`] from a [`LlamaGrammar`].
    ///
    /// `start_position` indicates the token position to begin applying the grammar at. [`None`]
    /// indicates that the grammar begins at the end of context.
    pub fn from_grammar(grammar: LlamaGrammar, start_position: Option<usize>) -> Self {
        SamplerStage::Grammar(GrammarStage {
            grammar,
            accepted_up_to: start_position,
        })
    }

    /// Applies this [`SamplerStage`] to the provided token data array.
    ///
    /// Ensures that at least `min_keep` tokens remain after the
    /// [`SamplerStage`]'s are applied.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn apply(
        &mut self,
        context: *mut llama_context,
        tokens: &[Token],
        mut candidates_p: llama_token_data_array,
        min_keep: usize,
    ) -> llama_token_data_array {
        #[cfg(feature = "legacy_sampler_compatibility")]
        {
            unimplemented!("The old sampling API has been deprecated in llama.cpp. Use sample_new() or the SamplerChain API directly.")
        }
        
        #[cfg(not(feature = "legacy_sampler_compatibility"))]
        {
            let _ = (context, tokens, min_keep, self);
            // Without the legacy compatibility feature, this method shouldn't exist
            // but we need it for the trait implementation
            unimplemented!("Legacy sampling not available. Use sample_new() or SamplerChain API.")
        }
    }
}

/// Opaque internals for [`SamplerStage::Grammar`].
#[derive(Clone, Debug)]
pub struct GrammarStage {
    grammar: LlamaGrammar,
    accepted_up_to: Option<usize>,
}

impl GrammarStage {
    fn apply(
        &mut self,
        context: *mut llama_context,
        tokens: &[Token],
        mut candidates_p: llama_token_data_array,
        _min_keep: usize,
    ) -> llama_token_data_array {
        let _ = (context, tokens, self);
        unimplemented!("Grammar API has been deprecated in llama.cpp. Use the new sampler chain with grammar sampler.")
    }
}

/// Determines how the next token is selected from the distribution produced by
/// the model and the [`SamplerStage`]'s.
#[derive(Clone, Debug)]
#[non_exhaustive]
enum TokenSelector {
    /// Selects a token at random, weighted by the distribution
    Softmax,

    /// Always selects the most likely token.
    Greedy,

    /// Selects a token using [Mirostat](https://arxiv.org/pdf/2007.14966.pdf)
    Mirostat { tau: f32, eta: f32, m: i32, mu: f32 },

    /// Selects a token using [Mirostat V2](https://arxiv.org/pdf/2007.14966.pdf)
    MirostatV2 { tau: f32, eta: f32, mu: f32 },
}

impl TokenSelector {
    /// Select and and return a token from a given distribution.
    ///
    /// Note: while this function may take a mutable reference to `self`, the internal state *shouldn't* be altered.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn select(
        &mut self,
        context: *mut llama_context,
        mut candidates_p: llama_token_data_array,
    ) -> Token {
        #[cfg(feature = "legacy_sampler_compatibility")]
        {
            unimplemented!("The old token selection API has been deprecated in llama.cpp. Use the SamplerChain API.")
        }
        
        #[cfg(not(feature = "legacy_sampler_compatibility"))]
        {
            let _ = (context, candidates_p, self);
            unimplemented!("Legacy token selection not available. Use SamplerChain API.")
        }
    }
}

/// Selects a token after applying multiple [`SamplerStage`]'s to the
/// probability distribution output by the model.
#[derive(Clone, Debug)]
pub struct StandardSampler {
    stages: Vec<SamplerStage>,
    min_keep: usize,
    token_selector: TokenSelector,
}

impl StandardSampler {
    /// Creates a new [`StandardSampler`] that modifies the model's raw
    /// distribution using multiple [`SamplerStage`]'s, then selects a random
    /// token from that distrubution.
    ///
    /// Ensures that at least `min_keep` tokens remain after the
    /// [`SamplerStage`]'s are applied.
    pub fn new_softmax(stages: Vec<SamplerStage>, min_keep: usize) -> StandardSampler {
        StandardSampler {
            stages,
            min_keep,
            token_selector: TokenSelector::Softmax,
        }
    }
    
    /// Convert the StandardSampler configuration to a new sampler chain
    fn to_sampler_chain(&self, context: *mut llama_context, seed: u32) -> Result<SamplerChain, SamplerChainError> {
        let mut chain = SamplerChain::new()?;
        
        // Add stages in order
        for stage in &self.stages {
            match stage {
                SamplerStage::RepetitionPenalty { 
                    repetition_penalty, 
                    frequency_penalty,
                    presence_penalty, 
                    last_n 
                } => {
                    chain.add(SamplerType::Penalties {
                        tokens: vec![],  // Will be populated during sampling
                        penalty_last_n: *last_n,
                        penalty_repeat: *repetition_penalty,
                        penalty_freq: *frequency_penalty,
                        penalty_present: *presence_penalty,
                    })?;
                }
                SamplerStage::Temperature(t) => {
                    if *t == 0.0 {
                        // Temperature 0 means greedy sampling
                        chain.add(SamplerType::TopK { k: 1 })?;
                    } else {
                        chain.add(SamplerType::Temperature { t: *t })?;
                    }
                }
                SamplerStage::DynamicTemperature { min_temp, max_temp, exponent_val } => {
                    chain.add(SamplerType::TempExt { 
                        t: *min_temp, 
                        delta: *max_temp - *min_temp, 
                        exponent: *exponent_val 
                    })?;
                }
                SamplerStage::TopK(k) => {
                    chain.add(SamplerType::TopK { k: *k })?;
                }
                SamplerStage::TopP(p) => {
                    chain.add(SamplerType::TopP { p: *p, min_keep: self.min_keep })?;
                }
                SamplerStage::MinP(p) => {
                    chain.add(SamplerType::MinP { p: *p, min_keep: self.min_keep })?;
                }
                SamplerStage::Typical(p) => {
                    chain.add(SamplerType::Typical { p: *p, min_keep: self.min_keep })?;
                }
                SamplerStage::TailFree(_z) => {
                    // TailFree is not supported in the new API, skip it
                    // TODO: Log a warning or handle this case differently
                }
                SamplerStage::Grammar(stage) => {
                    // Get the model from the context to create the grammar sampler
                    let model = unsafe { llama_get_model(context) };
                    let sampler_type = stage.grammar.to_sampler_type(model);
                    chain.add(sampler_type)?;
                }
            }
        }
        
        // Add final token selector
        match &self.token_selector {
            TokenSelector::Greedy => chain.add(SamplerType::Greedy)?,
            TokenSelector::Softmax => chain.add(SamplerType::Dist { seed })?,
            TokenSelector::Mirostat { tau, eta, m, mu: _ } => {
                chain.add(SamplerType::Mirostat { tau: *tau, eta: *eta, m: *m })?;
            }
            TokenSelector::MirostatV2 { tau, eta, mu: _ } => {
                chain.add(SamplerType::MirostatV2 { tau: *tau, eta: *eta })?;
            }
        }
        
        Ok(chain)
    }

    /// Creates a new [`StandardSampler`] that always selects the next most
    /// token produced by the model.
    pub fn new_greedy() -> StandardSampler {
        StandardSampler {
            stages: Vec::new(),
            min_keep: 0,
            token_selector: TokenSelector::Greedy,
        }
    }

    /// Creates a new [`StandardSampler`] that selects a token using
    /// [Mirostat](https://arxiv.org/pdf/2007.14966.pdf).
    pub fn new_mirostat(
        stages: Vec<SamplerStage>,
        min_keep: usize,
        tau: f32,
        eta: f32,
        m: i32,
    ) -> StandardSampler {
        StandardSampler {
            stages,
            min_keep,
            token_selector: TokenSelector::Mirostat {
                tau,
                eta,
                m,
                mu: 2.0 * tau,
            },
        }
    }

    /// Creates a new [`StandardSampler`] that selects a token using
    /// [Mirostat V2](https://arxiv.org/pdf/2007.14966.pdf).
    pub fn new_mirostat_v2(
        stages: Vec<SamplerStage>,
        min_keep: usize,
        tau: f32,
        eta: f32,
    ) -> StandardSampler {
        StandardSampler {
            stages,
            min_keep,
            token_selector: TokenSelector::MirostatV2 {
                tau,
                eta,
                mu: 2.0 * tau,
            },
        }
    }
    
    /// Sample using the new sampler chain API
    pub fn sample_new(
        &mut self,
        context: *mut llama_context,
        tokens: &[Token],
        _candidates_p: llama_token_data_array,
    ) -> Token {
        // Use a fixed seed for now, could be made configurable
        let mut chain = self.to_sampler_chain(context, 42).expect("Failed to create sampler chain");
        
        // Accept previous tokens for context
        for token in tokens {
            chain.accept(*token);
        }
        
        // Sample next token
        let token = chain.sample(context, -1);
        chain.accept(token);
        
        token
    }
}

impl Default for StandardSampler {
    fn default() -> Self {
        Self {
            stages: vec![
                SamplerStage::RepetitionPenalty {
                    repetition_penalty: 1.1,
                    frequency_penalty: 0.0,
                    presence_penalty: 0.0,
                    last_n: 64,
                },
                SamplerStage::TopK(40),
                SamplerStage::TopP(0.95),
                SamplerStage::MinP(0.05),
                SamplerStage::Temperature(0.8),
            ],
            min_keep: 1,
            token_selector: TokenSelector::Softmax,
        }
    }
}

impl Sampler for StandardSampler {
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn sample(
        &mut self,
        context: *mut llama_context,
        tokens: &[Token],
        mut candidates_p: llama_token_data_array,
    ) -> Token {
        let min_keep = self.min_keep.max(1);

        for stage in &mut self.stages {
            candidates_p = stage.apply(context, tokens, candidates_p, min_keep);
        }

        self.token_selector.select(context, candidates_p)
    }
}
