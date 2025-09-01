//! Implements [`LlamaParams`]

use std::ptr;

use llama_cpp_sys::{
    llama_context_default_params, llama_context_params, llama_model_default_params,
    llama_model_params, llama_pooling_type, llama_split_mode,
};

/// Parameters for llama.
pub struct LlamaParams {
    /// Number of layers to store in VRAM.
    ///
    /// If this number is bigger than the amount of model layers, all layers are loaded to VRAM.
    pub n_gpu_layers: u32,

    /// How to split the model across multiple GPUs
    pub split_mode: SplitMode,

    /// The GPU that is used for scratch and small tensors
    pub main_gpu: u32,

    /// How to split layers across multiple GPUs (size: LLAMA_MAX_DEVICES)
    //const float * tensor_split, TODO

    /// Called with a progress value between 0 and 1, pass NULL to disable
    //llama_progress_callback progress_callback, TODO

    /// Context pointer passed to the progress callback
    //void * progress_callback_user_data, TODO

    /// Override key-value pairs of the model meta data
    //const struct llama_model_kv_override * kv_overrides, TODO

    /// Only load the vocabulary, no weights
    pub vocab_only: bool,

    /// Use mmap if possible
    pub use_mmap: bool,

    /// Force system to keep model in RAM
    pub use_mlock: bool,

    /// Pooling strategy for embedding models
    pooling_type: EmbeddingModelPoolingType,
}

/// A policy to split the model across multiple GPUs
#[non_exhaustive]
pub enum SplitMode {
    /// Single GPU.
    ///
    /// Equivalent to [`llama_split_mode_LLAMA_SPLIT_NONE`]
    None,

    /// Split layers and KV across GPUs
    ///
    /// Equivalent to [`llama_split_mode_LLAMA_SPLIT_LAYER`]
    Layer,

    /// Split rows across GPUs
    ///
    /// Equivalent to [`llama_split_mode_LLAMA_SPLIT_ROW`]
    Row,
}

/// Pooling strategy for embedding models
///
/// Different models support different pooling strategies:
/// - BERT models typically use CLS pooling
/// - Sentence transformers often use Mean pooling
/// - Some models work best with no pooling
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmbeddingModelPoolingType {
    /// Let the model decide (default behavior)
    Unspecified,
    /// No pooling - return raw token embeddings
    None,
    /// Mean pooling - average all token embeddings
    Mean,
    /// CLS pooling - use only the [CLS] token embedding
    CLS,
}

impl From<SplitMode> for llama_split_mode {
    fn from(value: SplitMode) -> Self {
        match value {
            SplitMode::None => llama_split_mode::LLAMA_SPLIT_MODE_NONE,
            SplitMode::Layer => llama_split_mode::LLAMA_SPLIT_MODE_LAYER,
            SplitMode::Row => llama_split_mode::LLAMA_SPLIT_MODE_ROW,
        }
    }
}

impl From<llama_split_mode> for SplitMode {
    fn from(value: llama_split_mode) -> Self {
        #![allow(non_upper_case_globals)]
        match value {
            llama_split_mode::LLAMA_SPLIT_MODE_NONE => SplitMode::None,
            llama_split_mode::LLAMA_SPLIT_MODE_LAYER => SplitMode::Layer,
            llama_split_mode::LLAMA_SPLIT_MODE_ROW => SplitMode::Row,
            _ => unimplemented!(),
        }
    }
}

impl From<EmbeddingModelPoolingType> for llama_pooling_type {
    fn from(value: EmbeddingModelPoolingType) -> Self {
        match value {
            EmbeddingModelPoolingType::Unspecified => {
                llama_pooling_type::LLAMA_POOLING_TYPE_UNSPECIFIED
            }
            EmbeddingModelPoolingType::None => llama_pooling_type::LLAMA_POOLING_TYPE_NONE,
            EmbeddingModelPoolingType::Mean => llama_pooling_type::LLAMA_POOLING_TYPE_MEAN,
            EmbeddingModelPoolingType::CLS => llama_pooling_type::LLAMA_POOLING_TYPE_CLS,
        }
    }
}

impl From<llama_pooling_type> for EmbeddingModelPoolingType {
    fn from(value: llama_pooling_type) -> Self {
        match value {
            llama_pooling_type::LLAMA_POOLING_TYPE_UNSPECIFIED => {
                EmbeddingModelPoolingType::Unspecified
            }
            llama_pooling_type::LLAMA_POOLING_TYPE_NONE => EmbeddingModelPoolingType::None,
            llama_pooling_type::LLAMA_POOLING_TYPE_MEAN => EmbeddingModelPoolingType::Mean,
            llama_pooling_type::LLAMA_POOLING_TYPE_CLS => EmbeddingModelPoolingType::CLS,
            _ => unimplemented!(),
        }
    }
}

impl Default for LlamaParams {
    fn default() -> Self {
        // SAFETY: Stack constructor, always safe
        let c_params = unsafe { llama_model_default_params() };

        Self {
            n_gpu_layers: c_params.n_gpu_layers as u32,
            split_mode: c_params.split_mode.into(),
            main_gpu: c_params.main_gpu as u32,
            vocab_only: c_params.vocab_only,
            use_mmap: c_params.use_mmap,
            use_mlock: c_params.use_mlock,
            pooling_type: EmbeddingModelPoolingType::Unspecified,
        }
    }
}

impl From<LlamaParams> for llama_model_params {
    fn from(value: LlamaParams) -> Self {
        llama_model_params {
            devices: ptr::null_mut(),  // NULL-terminated device list
            tensor_buft_overrides: ptr::null(),  // Tensor buffer overrides
            n_gpu_layers: value.n_gpu_layers as i32,
            split_mode: value.split_mode.into(),
            main_gpu: value.main_gpu as i32,
            tensor_split: ptr::null_mut(),
            progress_callback: None,
            progress_callback_user_data: ptr::null_mut(),
            kv_overrides: ptr::null_mut(),
            vocab_only: value.vocab_only,
            use_mmap: value.use_mmap,
            use_mlock: value.use_mlock,
            check_tensors: false,  // Don't validate tensor checksums by default
            use_extra_bufts: false,  // Don't use extra buffers by default
        }
    }
}

/// Embeddings inference specific parameters.
pub struct EmbeddingsParams {
    /// number of threads to use for generation
    pub n_threads: u32,

    /// number of threads to use for batch processing
    pub n_threads_batch: u32,

    /// Type of pooling (if any) to use in embegging
    pub pooling_type: EmbeddingModelPoolingType,
}

impl EmbeddingsParams {
    pub(crate) fn as_context_params(&self, batch_capacity: usize) -> llama_context_params {
        // SAFETY: Stack constructor, always safe.
        let mut ctx_params = unsafe { llama_context_default_params() };

        ctx_params.embeddings = true;
        ctx_params.n_threads = self.n_threads as i32;
        ctx_params.n_threads_batch = self.n_threads_batch as i32;
        ctx_params.n_ctx = batch_capacity as u32;
        ctx_params.n_batch = batch_capacity as u32;
        ctx_params.n_ubatch = batch_capacity as u32;
        ctx_params.pooling_type = self.pooling_type.into();

        ctx_params
    }

    /// Create a new builder for configuring embedding parameters
    pub fn builder() -> EmbeddingsParamsBuilder {
        EmbeddingsParamsBuilder::default()
    }
}

impl Default for EmbeddingsParams {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl EmbeddingsParams {}

/// Builder for configuring embedding parameters
pub struct EmbeddingsParamsBuilder {
    n_threads: u32,
    n_threads_batch: u32,
    pooling_type: EmbeddingModelPoolingType,
}

impl Default for EmbeddingsParamsBuilder {
    fn default() -> Self {
        let threads = num_cpus::get_physical() as u32 - 1;
        Self {
            n_threads: threads,
            n_threads_batch: threads,
            pooling_type: EmbeddingModelPoolingType::Unspecified,
        }
    }
}

impl EmbeddingsParamsBuilder {
    /// Set the number of threads to use for generation
    pub fn n_threads(mut self, threads: u32) -> Self {
        self.n_threads = threads;
        self
    }

    /// Set the number of threads to use for batch processing
    pub fn n_threads_batch(mut self, threads: u32) -> Self {
        self.n_threads_batch = threads;
        self
    }

    /// Set the pooling type for embeddings
    pub fn pooling_type(mut self, pooling: EmbeddingModelPoolingType) -> Self {
        self.pooling_type = pooling;
        self
    }

    /// Build the final EmbeddingsParams
    pub fn build(self) -> EmbeddingsParams {
        EmbeddingsParams {
            n_threads: self.n_threads,
            n_threads_batch: self.n_threads_batch,
            pooling_type: self.pooling_type,
        }
    }
}
