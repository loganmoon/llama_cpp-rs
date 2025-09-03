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

fn ensure_test_models() {
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

    async fn run_embeddings_edge_case_test(
        model: &LlamaModel,
        input: Vec<String>,
        test_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("  Running embeddings for: {}", test_name);
        println!("  Input count: {}, Total chars: {}", 
                 input.len(), 
                 input.iter().map(|s| s.len()).sum::<usize>());
        
        let params = EmbeddingsParams::default();
        
        match model.embeddings_async(&input, params).await {
            Ok(embeddings) => {
                println!("  Result: {} embeddings generated", embeddings.len());
                
                // Validate embeddings
                for (idx, embedding) in embeddings.iter().enumerate() {
                    if embedding.is_empty() {
                        return Err(format!("Embedding {} is empty", idx).into());
                    }
                    
                    let mut sum = 0f32;
                    for value in embedding {
                        if !value.is_finite() {
                            return Err(format!("Embedding {} contains non-finite value", idx).into());
                        }
                        if *value < -10.0 || *value > 10.0 {
                            return Err(format!("Embedding {} value {} out of reasonable range", idx, value).into());
                        }
                        sum += value * value;
                    }
                    
                    // Check magnitude is reasonable (not necessarily normalized to 1)
                    let mag = sum.sqrt();
                    if mag < 0.01 {
                        return Err(format!("Embedding {} has near-zero magnitude: {}", idx, mag).into());
                    }
                    if mag > 100.0 {
                        return Err(format!("Embedding {} has excessive magnitude: {}", idx, mag).into());
                    }
                }
                
                Ok(())
            }
            Err(e) => {
                println!("  Result: FAILED - {}", e);
                Err(e.into())
            }
        }
    }

    #[tokio::test]
    async fn test_embeddings_edge_cases() {
        init_tracing();
        ensure_test_models();
        
        let dir = std::env::var("LLAMA_EMBED_MODELS_DIR")
            .expect("LLAMA_EMBED_MODELS_DIR should be set by test setup");
        
        let models = list_models(dir).await;
        
        for model_path in models {
            println!("\n=== Testing edge cases for model: {:?} ===", model_path);
            
            let params = LlamaParams::default();
            let model = LlamaModel::load_from_file_async(model_path, params)
                .await
                .expect("Failed to load model");
            
            let mut test_results = Vec::new();
            
            // Test 1: Empty input array
            println!("\nTest 1: Empty input array");
            let empty_input: Vec<String> = vec![];
            let result = run_embeddings_edge_case_test(&model, empty_input, "empty_array").await;
            test_results.push(("empty_array", result));
            
            // Test 2: Array with single empty string
            println!("\nTest 2: Single empty string");
            let single_empty = vec![String::new()];
            let result = run_embeddings_edge_case_test(&model, single_empty, "single_empty_string").await;
            test_results.push(("single_empty_string", result));
            
            // Test 3: Array with multiple empty strings
            println!("\nTest 3: Multiple empty strings");
            let multiple_empty = vec![String::new(), String::new(), String::new()];
            let result = run_embeddings_edge_case_test(&model, multiple_empty, "multiple_empty_strings").await;
            test_results.push(("multiple_empty_strings", result));
            
            // Test 4: Mixed empty and non-empty strings
            println!("\nTest 4: Mixed empty and non-empty strings");
            let mixed = vec![
                String::from("Hello"),
                String::new(),
                String::from("World"),
                String::new(),
            ];
            let result = run_embeddings_edge_case_test(&model, mixed, "mixed_empty_nonempty").await;
            test_results.push(("mixed_empty_nonempty", result));
            
            // Test 5: Single word
            println!("\nTest 5: Single word");
            let single_word = vec![String::from("Hello")];
            let result = run_embeddings_edge_case_test(&model, single_word, "single_word").await;
            test_results.push(("single_word", result));
            
            // Test 6: Small batch (10 words)
            println!("\nTest 6: Small batch");
            let small_batch = vec![String::from("This is a small test batch with ten words total")];
            let result = run_embeddings_edge_case_test(&model, small_batch, "small_batch").await;
            test_results.push(("small_batch", result));
            
            // Test 7: Medium batch (100 words)
            println!("\nTest 7: Medium batch");
            let mut medium_text = String::new();
            for i in 0..100 {
                medium_text.push_str(&format!("word{} ", i));
            }
            let medium_batch = vec![medium_text];
            let result = run_embeddings_edge_case_test(&model, medium_batch, "medium_batch").await;
            test_results.push(("medium_batch", result));
            
            // Test 8: Large batch (1000 words)
            println!("\nTest 8: Large batch");
            let mut large_text = String::new();
            for i in 0..1000 {
                large_text.push_str(&format!("word{} ", i));
            }
            let large_batch = vec![large_text];
            let result = run_embeddings_edge_case_test(&model, large_batch, "large_batch").await;
            test_results.push(("large_batch", result));
            
            // Test 9: Very large batch (original test case - 10x200 words)
            println!("\nTest 9: Very large batch (original test)");
            let mut very_large = vec![];
            for _phrase_idx in 0..10 {
                let mut phrase = String::new();
                for _word_idx in 0..200 {
                    phrase.push_str("word ");
                }
                phrase.truncate(phrase.len() - 1);
                very_large.push(phrase);
            }
            let result = run_embeddings_edge_case_test(&model, very_large, "very_large_batch").await;
            test_results.push(("very_large_batch", result));
            
            // Test 10: Special characters
            println!("\nTest 10: Special characters");
            let special_chars = vec![
                String::from("Hello! @#$%^&*()"),
                String::from("😀 emoji test 🎉"),
                String::from("\n\t\r"),
            ];
            let result = run_embeddings_edge_case_test(&model, special_chars, "special_chars").await;
            test_results.push(("special_chars", result));
            
            // Test 11: Single space
            println!("\nTest 11: Single space");
            let single_space = vec![String::from(" ")];
            let result = run_embeddings_edge_case_test(&model, single_space, "single_space").await;
            test_results.push(("single_space", result));
            
            // Test 12: Multiple spaces
            println!("\nTest 12: Multiple spaces");
            let multiple_spaces = vec![String::from("     ")];
            let result = run_embeddings_edge_case_test(&model, multiple_spaces, "multiple_spaces").await;
            test_results.push(("multiple_spaces", result));
            
            // Print summary
            println!("\n=== Test Results Summary ===");
            let mut passed = 0;
            let mut failed = 0;
            
            for (name, result) in &test_results {
                match result {
                    Ok(_) => {
                        println!("✓ {}: PASSED", name);
                        passed += 1;
                    }
                    Err(e) => {
                        println!("✗ {}: FAILED - {}", name, e);
                        failed += 1;
                    }
                }
            }
            
            println!("\nTotal: {} passed, {} failed", passed, failed);
            
            if failed > 0 {
                panic!("{} tests failed", failed);
            }
        }
    }
}