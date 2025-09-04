//! Comprehensive edge case tests for embeddings to ensure failures happen in Rust wrapper, not FFI
//! 
//! These tests cover critical edge cases that could cause crashes or undefined behavior
//! if not properly handled before the FFI boundary.

use llama_cpp::{EmbeddingsParams, LlamaModel, LlamaParams};
use std::sync::Arc;

/// Helper function to load the first available embedding model
pub(crate) async fn load_test_embedding_model() -> Result<LlamaModel, Box<dyn std::error::Error>> {
    let dir = std::env::var("LLAMA_EMBED_MODELS_DIR")
        .expect("LLAMA_EMBED_MODELS_DIR should be set by test setup");
    
    let mut entries = tokio::fs::read_dir(&dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
            let model = LlamaModel::load_from_file_async(
                path.to_str().unwrap(),
                LlamaParams::default()
            ).await?;
            return Ok(model);
        }
    }
    
    Err("No embedding model found".into())
}

/// Test inputs that might tokenize to zero tokens
#[tokio::test]
async fn test_tokenization_edge_cases() {
    crate::ensure_test_models();
    
    let model = load_test_embedding_model().await
        .expect("Failed to load embedding model");
    
    let test_cases = vec![
        // Strings that might produce no tokens or unusual tokenization
        ("zero_width_chars", vec![
            String::from("\u{200B}"), // Zero-width space
            String::from("\u{200C}"), // Zero-width non-joiner
            String::from("\u{200D}"), // Zero-width joiner
            String::from("\u{FEFF}"), // Zero-width no-break space
        ]),
        ("control_chars", vec![
            String::from("\x00"), // Null byte
            String::from("\x01\x02\x03"), // Control characters
            String::from("\x1B[31m"), // ANSI escape sequence
            String::from("\r\n\t\x0B\x0C"), // Various whitespace control chars
        ]),
        ("invalid_utf8_recovery", vec![
            // These are valid UTF-8 but might cause issues in tokenization
            String::from("\u{FFFD}"), // Replacement character
        ]),
        ("boundary_chars", vec![
            String::from("\u{10FFFF}"), // Maximum valid Unicode code point
            String::from("\u{0000}"), // Minimum Unicode (null)
        ]),
    ];
    
    let _params = EmbeddingsParams::default();
    
    for (test_name, inputs) in test_cases {
        println!("\nTesting tokenization edge case: {}", test_name);
        
        match model.embeddings_async(&inputs, EmbeddingsParams::default()).await {
            Ok(embeddings) => {
                println!("  ✓ Successfully generated {} embeddings", embeddings.len());
                assert_eq!(embeddings.len(), inputs.len(), 
                    "Number of embeddings should match number of inputs");
                
                // Verify each embedding is valid
                for (idx, embedding) in embeddings.iter().enumerate() {
                    assert!(!embedding.is_empty() || inputs[idx].is_empty(), 
                        "Non-empty input should produce non-empty embedding");
                    
                    for value in embedding {
                        assert!(value.is_finite(), 
                            "Embedding values should be finite numbers");
                    }
                }
            }
            Err(e) => {
                println!("  ✓ Properly caught error in Rust wrapper: {}", e);
                // This is acceptable - we want errors caught in Rust, not in FFI
            }
        }
    }
}

/// Test batch boundary conditions
#[tokio::test] 
async fn test_batch_boundary_cases() {
    crate::ensure_test_models();
    
    let model = load_test_embedding_model().await
        .expect("Failed to load embedding model");
    
    // Estimate a reasonable batch size based on model
    // We'll create inputs that stress the batching logic
    
    // Test 1: Many small inputs that require multiple batches
    println!("\nTest: Many small inputs requiring batching");
    let many_small: Vec<String> = (0..100)
        .map(|i| format!("Input number {}", i))
        .collect();
    
    match model.embeddings_async(&many_small, EmbeddingsParams::default()).await {
        Ok(embeddings) => {
            println!("  ✓ Successfully processed {} small inputs", embeddings.len());
            assert_eq!(embeddings.len(), many_small.len());
        }
        Err(e) => {
            println!("  ✓ Caught error: {}", e);
        }
    }
    
    // Test 2: Single very large input that might exceed batch capacity
    println!("\nTest: Single large input");
    let large_input = vec!["word ".repeat(10000)];
    
    match model.embeddings_async(&large_input, EmbeddingsParams::default()).await {
        Ok(embeddings) => {
            println!("  ✓ Successfully processed large input");
            assert_eq!(embeddings.len(), 1);
        }
        Err(e) => {
            println!("  ✓ Caught error for oversized input: {}", e);
        }
    }
    
    // Test 3: Mixed sizes where one input causes batch split
    println!("\nTest: Mixed input sizes");
    let mixed = vec![
        "Short".to_string(),
        "word ".repeat(5000), // Large input
        "Another short one".to_string(),
    ];
    
    match model.embeddings_async(&mixed, EmbeddingsParams::default()).await {
        Ok(embeddings) => {
            println!("  ✓ Successfully processed mixed batch");
            assert_eq!(embeddings.len(), mixed.len());
        }
        Err(e) => {
            println!("  ✓ Caught error: {}", e);
        }
    }
}

/// Test numerical stability edge cases
#[tokio::test]
async fn test_numerical_stability() {
    crate::ensure_test_models();
    
    let model = load_test_embedding_model().await
        .expect("Failed to load embedding model");
    
    // Test various numerical edge cases
    let test_cases = vec![
        ("repeated_token", vec![
            "a".repeat(1000), // Single character repeated many times
        ]),
        ("unicode_math_symbols", vec![
            String::from("∑∏∫∂∇×÷±∞"), // Mathematical symbols
        ]),
        ("extreme_unicode", vec![
            String::from("𝕳𝖊𝖑𝖑𝖔 𝖂𝖔𝖗𝖑𝖉"), // Mathematical bold text
            String::from("🔢🔣🔤"), // Unicode symbols
        ]),
        ("mixed_scripts", vec![
            String::from("Hello مرحبا 你好 こんにちは שלום"),
        ]),
    ];
    
    for (test_name, inputs) in test_cases {
        println!("\nTesting numerical stability: {}", test_name);
        
        match model.embeddings_async(&inputs, EmbeddingsParams::default()).await {
            Ok(embeddings) => {
                for (idx, embedding) in embeddings.iter().enumerate() {
                    // Check for numerical validity
                    let mut sum = 0f32;
                    let mut min = f32::MAX;
                    let mut max = f32::MIN;
                    
                    for &value in embedding {
                        assert!(value.is_finite(), 
                            "Embedding {} contains non-finite value", idx);
                        sum += value * value;
                        min = min.min(value);
                        max = max.max(value);
                    }
                    
                    let magnitude = sum.sqrt();
                    println!("  Embedding {}: magnitude={:.4}, range=[{:.4}, {:.4}]", 
                        idx, magnitude, min, max);
                    
                    // Check magnitude is reasonable (not zero, not infinite)
                    assert!(magnitude > 0.0, "Embedding magnitude should be non-zero");
                    assert!(magnitude.is_finite(), "Embedding magnitude should be finite");
                    
                    // Check range is reasonable
                    assert!(min > -1000.0 && max < 1000.0, 
                        "Embedding values should be in reasonable range");
                }
                println!("  ✓ All embeddings numerically stable");
            }
            Err(e) => {
                println!("  ✓ Caught error: {}", e);
            }
        }
    }
}

/// Test concurrent access to embeddings
#[tokio::test]
async fn test_concurrent_embeddings() {
    crate::ensure_test_models();
    
    let model = Arc::new(load_test_embedding_model().await
        .expect("Failed to load embedding model"));
    
    // Launch multiple concurrent embedding tasks
    let mut tasks = vec![];
    
    for i in 0..5 {
        let model_clone = model.clone();
        let task = tokio::spawn(async move {
            let inputs = vec![
                format!("Task {} input 1", i),
                format!("Task {} input 2", i),
                format!("Task {} input 3", i),
            ];
            
            model_clone.embeddings_async(&inputs, EmbeddingsParams::default()).await
        });
        tasks.push(task);
    }
    
    // Wait for all tasks and check results
    let mut successes = 0;
    let mut errors = 0;
    
    for (i, task) in tasks.into_iter().enumerate() {
        match task.await {
            Ok(Ok(embeddings)) => {
                println!("Task {} succeeded with {} embeddings", i, embeddings.len());
                successes += 1;
                
                // Verify embeddings are valid
                for embedding in &embeddings {
                    assert!(!embedding.is_empty(), "Embedding should not be empty");
                    for &value in embedding {
                        assert!(value.is_finite(), "Values should be finite");
                    }
                }
            }
            Ok(Err(e)) => {
                println!("Task {} failed with error: {}", i, e);
                errors += 1;
            }
            Err(e) => {
                println!("Task {} panicked: {}", i, e);
                errors += 1;
            }
        }
    }
    
    println!("\nConcurrent test results: {} successes, {} errors", successes, errors);
    
    // At least some tasks should succeed
    assert!(successes > 0, "At least some concurrent tasks should succeed");
}

/// Test empty and whitespace-only inputs specifically
#[tokio::test]
async fn test_empty_and_whitespace() {
    crate::ensure_test_models();
    
    let model = load_test_embedding_model().await
        .expect("Failed to load embedding model");
    
    let test_cases = vec![
        ("empty_string", vec![String::new()]),
        ("single_space", vec![String::from(" ")]),
        ("multiple_spaces", vec![String::from("     ")]),
        ("tabs", vec![String::from("\t\t\t")]),
        ("newlines", vec![String::from("\n\n\n")]),
        ("mixed_whitespace", vec![String::from(" \t \n \r ")]),
        ("zero_width_space", vec![String::from("\u{200B}")]),
    ];
    
    for (test_name, inputs) in test_cases {
        println!("\nTesting: {}", test_name);
        
        match model.embeddings_async(&inputs, EmbeddingsParams::default()).await {
            Ok(embeddings) => {
                println!("  ✓ Generated {} embeddings", embeddings.len());
                assert_eq!(embeddings.len(), inputs.len());
                
                // For empty/whitespace inputs, we expect either:
                // 1. Zero embeddings (all zeros)
                // 2. Some default embedding
                // Either is fine as long as it doesn't crash
                
                for embedding in &embeddings {
                    if !embedding.is_empty() {
                        for &value in embedding {
                            assert!(value.is_finite(), "Values should be finite");
                        }
                    }
                }
            }
            Err(e) => {
                println!("  ✓ Properly handled error: {}", e);
            }
        }
    }
}

/// Test mixed valid and edge case inputs in single batch
#[tokio::test]
async fn test_mixed_edge_cases() {
    crate::ensure_test_models();
    
    let model = load_test_embedding_model().await
        .expect("Failed to load embedding model");
    
    // Mix of normal and edge case inputs
    let inputs = vec![
        String::from("Normal text input"),
        String::new(), // Empty
        String::from("\0"), // Null byte
        String::from("Another normal input"),
        String::from("   "), // Spaces
        String::from("Final normal text"),
        String::from("\u{200B}"), // Zero-width space
    ];
    
    println!("\nTesting mixed normal and edge case inputs");
    
    match model.embeddings_async(&inputs, EmbeddingsParams::default()).await {
        Ok(embeddings) => {
            println!("  ✓ Successfully processed {} mixed inputs", embeddings.len());
            assert_eq!(embeddings.len(), inputs.len());
            
            for (idx, embedding) in embeddings.iter().enumerate() {
                println!("  Input {}: {} dims", idx, embedding.len());
                
                // Verify basic properties
                if !inputs[idx].is_empty() || embedding.len() > 0 {
                    for &value in embedding {
                        assert!(value.is_finite(), 
                            "Input {} produced non-finite values", idx);
                    }
                }
            }
        }
        Err(e) => {
            println!("  ✓ Caught error in mixed batch: {}", e);
        }
    }
}

/// Helper function for running edge case tests with detailed validation
pub(crate) async fn run_embeddings_edge_case_test(
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
                
                // For empty or whitespace-only inputs, zero magnitude is acceptable
                let is_empty_input = idx < input.len() && 
                    (input[idx].is_empty() || input[idx].trim().is_empty());
                
                if !is_empty_input && mag < 0.01 {
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

/// Comprehensive test of all embeddings edge cases
#[tokio::test]
async fn test_embeddings_edge_cases_comprehensive() {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    use std::sync::atomic::{AtomicBool, Ordering};
    
    // Initialize tracing (use try_init to avoid conflict)
    static SUBSCRIBER_SET: AtomicBool = AtomicBool::new(false);
    if !SUBSCRIBER_SET.swap(true, Ordering::SeqCst) {
        let format = tracing_subscriber::fmt::layer().compact();
        let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or(
            tracing_subscriber::EnvFilter::default()
                .add_directive(tracing_subscriber::filter::LevelFilter::INFO.into()),
        );

        // Use try_init to avoid panic if already initialized
        let _ = tracing_subscriber::registry()
            .with(format)
            .with(filter)
            .try_init();
    }
    
    crate::ensure_test_models();
    
    let dir = std::env::var("LLAMA_EMBED_MODELS_DIR")
        .expect("LLAMA_EMBED_MODELS_DIR should be set by test setup");
    
    // List models
    let mut models = Vec::new();
    let mut entries = tokio::fs::read_dir(&dir).await.unwrap();
    while let Some(entry) = entries.next_entry().await.unwrap() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
            models.push(path.to_string_lossy().to_string());
        }
    }
    
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