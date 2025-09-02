#[cfg(test)]
mod tests {
    use llama_cpp::{EmbeddingsParams, EmbeddingModelPoolingType};

    #[test]
    fn test_builder_pattern() {
        // Test builder with mean pooling
        let params = EmbeddingsParams::builder()
            .pooling_type(EmbeddingModelPoolingType::Mean)
            .build();
        assert_eq!(params.pooling_type, EmbeddingModelPoolingType::Mean);

        // Test builder with threads configuration and CLS pooling
        let params = EmbeddingsParams::builder()
            .n_threads(8)
            .n_threads_batch(4)
            .pooling_type(EmbeddingModelPoolingType::CLS)
            .build();
        assert_eq!(params.n_threads, 8);
        assert_eq!(params.n_threads_batch, 4);
        assert_eq!(params.pooling_type, EmbeddingModelPoolingType::CLS);

        // Test chaining overwrites previous pooling
        let params = EmbeddingsParams::builder()
            .pooling_type(EmbeddingModelPoolingType::CLS)
            .pooling_type(EmbeddingModelPoolingType::Mean)
            .build();
        assert_eq!(params.pooling_type, EmbeddingModelPoolingType::Mean);
    }

    #[test]
    fn test_builder_defaults() {
        let params = EmbeddingsParams::builder().build();
        assert!(params.n_threads > 0);
        assert!(params.n_threads_batch > 0);
        assert_eq!(params.pooling_type, EmbeddingModelPoolingType::Unspecified);
    }
}
