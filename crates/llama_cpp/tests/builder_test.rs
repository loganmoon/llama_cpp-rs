#[cfg(test)]
mod tests {
    use llama_cpp::EmbeddingsParams;

    #[test]
    fn test_builder_pattern() {
        // Test factory methods
        let params = EmbeddingsParams::with_mean_pooling();
        assert!(params.pooling_type.is_some());

        // Test builder with pooling
        let params = EmbeddingsParams::builder().with_mean_pooling().build();
        assert!(params.pooling_type.is_some());

        // Test builder with threads configuration
        let params = EmbeddingsParams::builder()
            .n_threads(8)
            .n_threads_batch(4)
            .with_cls_pooling()
            .build();
        assert_eq!(params.n_threads, 8);
        assert_eq!(params.n_threads_batch, 4);
        assert!(params.pooling_type.is_some());

        // Test chaining overwrites previous pooling
        let params = EmbeddingsParams::builder()
            .with_cls_pooling()
            .with_mean_pooling()
            .build();
        assert!(params.pooling_type.is_some());
    }

    #[test]
    fn test_builder_defaults() {
        let params = EmbeddingsParams::builder().build();
        assert!(params.n_threads > 0);
        assert!(params.n_threads_batch > 0);
        assert_eq!(params.pooling_type, None);
    }
}
