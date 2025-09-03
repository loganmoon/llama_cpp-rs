//! Test harness for [`llama_cpp`][lcpp] and [`llama_cpp_sys`].
//!
//! This crate automatically downloads small test models from Hugging Face for testing.
//! Models are cached in a temporary directory and reused across test runs.

pub mod test_model_downloader;
pub mod embeddings_edge_cases;

use test_model_downloader::TestModelGenerator;
use once_cell::sync::Lazy;
use std::sync::Once;

static INIT: Once = Once::new();

pub(crate) fn ensure_test_models() {
    INIT.call_once(|| {
        println!("Setting up test models...");
        // Simply set environment variables to point to existing models
        // The embedding model should already be downloaded at /tmp/llama_cpp_test_models
        let models_dir = "/tmp/llama_cpp_test_models";
        std::env::set_var("LLAMA_CPP_TEST_MODELS", models_dir);
        std::env::set_var("LLAMA_EMBED_MODELS_DIR", models_dir);
        println!("Test models directory set to: {}", models_dir);
    });
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::io::Write;
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use futures::StreamExt;
    use tokio::select;
    use tokio::time::Instant;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    use llama_cpp::standard_sampler::StandardSampler;
    use llama_cpp::{
        CompletionHandle, EmbeddingsParams, LlamaModel, LlamaParams, SessionParams, TokensToStrings,
    };
    
    use crate::ensure_test_models;

    fn init_tracing() {
        static SUBSCRIBER_SET: AtomicBool = AtomicBool::new(false);

        if !SUBSCRIBER_SET.swap(true, Ordering::SeqCst) {
            let format = tracing_subscriber::fmt::layer().compact();
            let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or(
                tracing_subscriber::EnvFilter::default()
                    .add_directive(tracing_subscriber::filter::LevelFilter::INFO.into()),
            );

            tracing_subscriber::registry()
                .with(format)
                .with(filter)
                .init();
        }
    }

    async fn list_models(dir: impl AsRef<Path>) -> Vec<String> {
        let dir = dir.as_ref();

        if !dir.is_dir() {
            panic!("\"{}\" is not a directory", dir.to_string_lossy());
        }

        let mut models = tokio::fs::read_dir(dir).await.unwrap();
        let mut rv = vec![];

        while let Some(model) = models.next_entry().await.unwrap() {
            let path = model.path();

            if path.is_file() {
                let path = path.to_str().unwrap();
                if path.ends_with(".gguf") {
                    rv.push(path.to_string());
                }
            }
        }

        rv
    }

    // TODO theres a concurrency issue with vulkan, look into it
    #[tokio::test]
    async fn test_tokenize_bytes() {
        ensure_test_models();
        
        let dir = std::env::var("LLAMA_CPP_TEST_MODELS")
            .expect("LLAMA_CPP_TEST_MODELS should be set by test setup");
        
        let models = list_models(dir).await;
        
        // Test tokenize_bytes with edge case inputs
        for model_path in models.iter().take(1) { // Test with first model only
            println!("Testing tokenize_bytes with model: {}", model_path);
            
            let model = LlamaModel::load_from_file_async(model_path, LlamaParams::default())
                .await
                .expect("Failed to load model");
            
            let test_inputs = vec![
                ("\u{200B}", "Zero-width space"),
                ("\u{200C}", "Zero-width non-joiner"),
                ("\u{200D}", "Zero-width joiner"),
                ("\u{FEFF}", "Zero-width no-break space"),
                ("normal text", "Normal text"),
                ("", "Empty string"),
                (" ", "Single space"),
            ];
            
            for (input, description) in test_inputs {
                let result = model.tokenize_bytes(input.as_bytes(), true, false);
                match result {
                    Ok(tokens) => {
                        println!("  {} ({:?}) => {} tokens", description, input, tokens.len());
                        // Verify tokens are valid
                        for token in &tokens {
                            assert!(token.0 >= 0, "Token should be non-negative");
                        }
                    }
                    Err(e) => {
                        println!("  {} ({:?}) => Error: {}", description, input, e);
                        // Empty strings may fail, which is acceptable
                        if !input.is_empty() {
                            panic!("Tokenization failed for non-empty input: {}", e);
                        }
                    }
                }
            }
        }
    }
    
    #[tokio::test]
    async fn load_models() {
        ensure_test_models();
        
        let dir = std::env::var("LLAMA_CPP_TEST_MODELS")
            .expect("LLAMA_CPP_TEST_MODELS should be set by test setup");

        let models = list_models(dir).await;

        for model in models {
            println!("Loading model: {}", model);
            let _model = LlamaModel::load_from_file_async(model, LlamaParams::default())
                .await
                .expect("Failed to load model");
        }
    }

    #[tokio::test]
    async fn execute_completions() {
        init_tracing();
        ensure_test_models();

        let dir = std::env::var("LLAMA_CPP_TEST_MODELS")
            .expect("LLAMA_CPP_TEST_MODELS should be set by test setup");

        let models = list_models(dir).await;

        for model in models {
            let mut params = LlamaParams::default();

            if cfg!(any(feature = "vulkan", feature = "cuda", feature = "metal")) {
                params.n_gpu_layers = i32::MAX as u32;
            }

            let model = LlamaModel::load_from_file_async(model, params)
                .await
                .expect("Failed to load model");

            let params = SessionParams {
                n_ctx: 2048,
                ..Default::default()
            };

            let estimate = model.estimate_session_size(&params);
            println!(
                "Predict chat session size: Host {}MB, Device {}MB",
                estimate.host_memory / 1024 / 1024,
                estimate.device_memory / 1024 / 1024,
            );

            let mut session = model
                .create_session(params)
                .expect("Failed to create session");

            println!(
                "Real chat session size: Host {}MB",
                session.memory_size() / 1024 / 1024
            );

            session
                .advance_context_async("<|SYSTEM|>You are a helpful assistant.")
                .await
                .unwrap();
            session
                .advance_context_async("<|USER|>Hello!")
                .await
                .unwrap();
            session
                .advance_context_async("<|ASSISTANT|>")
                .await
                .unwrap();

            let mut completions = session
                .start_completing_with(StandardSampler::default(), 1024)
                .expect("Failed to start completing")
                .into_strings();
            let timeout_by = Instant::now() + Duration::from_secs(500);

            println!();

            loop {
                select! {
                    _ = tokio::time::sleep_until(timeout_by) => {
                        break;
                    }
                    completion = <TokensToStrings<CompletionHandle> as StreamExt>::next(&mut completions) => {
                        if let Some(completion) = completion {
                            print!("{completion}");
                            let _ = io::stdout().flush();
                        } else {
                            break;
                        }
                        continue;
                    }
                }
            }
            println!();
            println!();
        }
    }

    #[tokio::test]
    async fn embed() {
        init_tracing();
        ensure_test_models();

        let dir = std::env::var("LLAMA_EMBED_MODELS_DIR")
            .expect("LLAMA_EMBED_MODELS_DIR should be set by test setup");

        let models = list_models(dir).await;

        for model in models {
            let params = LlamaParams::default();
            let model = LlamaModel::load_from_file_async(model, params)
                .await
                .expect("Failed to load model");

            let mut input = vec![];

            for _phrase_idx in 0..10 {
                let mut phrase = String::new();
                for _word_idx in 0..200 {
                    phrase.push_str("word ");
                }
                phrase.truncate(phrase.len() - 1);
                input.push(phrase);
            }

            let params = EmbeddingsParams::default();

            let tokenized_input = model
                .tokenize_slice(&input, true, false)
                .expect("Failed to tokenize input");
            let estimate = model.estimate_embeddings_session_size(&tokenized_input, &params);
            println!(
                "Predict embeddings session size: Host {}MB, Device {}MB",
                estimate.host_memory / 1024 / 1024,
                estimate.device_memory / 1024 / 1024,
            );

            let res = model
                .embeddings_async(&input, params)
                .await
                .expect("Failed to infer embeddings");

            println!("{:?}", res[0]);

            for embedding in &res {
                let mut sum = 0f32;
                for value in embedding {
                    assert!(value.is_normal(), "Embedding value isn't normal");
                    assert!(*value >= -1f32, "Embedding value isn't normalised");
                    assert!(*value <= 1f32, "Embedding value isn't normalised");
                    sum += value * value;
                }

                const ERROR: f32 = 0.0001;
                let mag = sum.sqrt();
                assert!(mag < 1. + ERROR, "Vector magnitude is not close to 1");
                assert!(mag > 1. - ERROR, "Vector magnitude is not close to 1");
            }
        }
    }

    #[tokio::test]
    async fn embed_with_session() {
        init_tracing();
        ensure_test_models();

        let dir = std::env::var("LLAMA_EMBED_MODELS_DIR")
            .expect("LLAMA_EMBED_MODELS_DIR should be set by test setup");

        let models = list_models(dir).await;
        
        // Test with first embedding model
        if let Some(model_path) = models.first() {
            let model_params = LlamaParams::default();
            
            let model = LlamaModel::load_from_file_async(model_path, model_params)
                .await
                .expect("Failed to load embedding model");

            // Create session for embeddings
            let session_params = SessionParams::for_embeddings();
            let session = model.create_session(session_params)
                .expect("Failed to create embeddings session");

            // Test inputs
            let inputs = vec![
                "The quick brown fox jumps over the lazy dog",
                "Machine learning is a subset of artificial intelligence",
                "Rust is a systems programming language",
            ];

            // Generate embeddings with session
            let embeddings = model
                .embeddings_with_session(&session, &inputs, EmbeddingsParams::default())
                .expect("Failed to generate embeddings with session");

            // Validate results
            assert_eq!(embeddings.len(), inputs.len());
            for (i, embedding) in embeddings.iter().enumerate() {
                assert!(!embedding.is_empty(), "Embedding {} is empty", i);
                
                // Check normalization
                let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!((magnitude - 1.0).abs() < 0.01, "Embedding {} not normalized", i);
            }

            // Test mode switching
            session.set_embeddings_mode(false);
            session.set_embeddings_mode(true);
            
            // Generate again to ensure mode switching works
            let embeddings2 = model
                .embeddings_with_session(&session, &inputs[..1], EmbeddingsParams::default())
                .expect("Failed after mode switch");
            
            assert_eq!(embeddings2.len(), 1);
        }
    }

    #[tokio::test]
    async fn embed_session_performance_comparison() {
        init_tracing();
        ensure_test_models();

        let dir = std::env::var("LLAMA_EMBED_MODELS_DIR")
            .expect("LLAMA_EMBED_MODELS_DIR should be set by test setup");

        let models = list_models(dir).await;
        
        if let Some(model_path) = models.first() {
            let model = LlamaModel::load_from_file_async(model_path, LlamaParams::default())
                .await
                .expect("Failed to load model");

            let inputs = vec!["Test input"; 10];
            let params = EmbeddingsParams::default();

            // Time temporary context approach
            let start_temp = std::time::Instant::now();
            for _ in 0..3 {
                let _ = model.embeddings(&inputs, params).expect("Temporary context failed");
            }
            let duration_temp = start_temp.elapsed();

            // Time session-based approach
            let session = model.create_session(SessionParams::for_embeddings())
                .expect("Failed to create session");
            
            let start_session = std::time::Instant::now();
            for _ in 0..3 {
                let _ = model.embeddings_with_session(&session, &inputs, params)
                    .expect("Session-based failed");
            }
            let duration_session = start_session.elapsed();

            println!("Temporary context: {:?}", duration_temp);
            println!("Session-based: {:?}", duration_session);
            
            // Session approach should be faster for repeated calls
            // We don't assert this as performance can vary, but log for manual verification
        }
    }

}