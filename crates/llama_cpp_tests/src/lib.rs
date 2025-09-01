//! Test harness for [`llama_cpp`][lcpp] and [`llama_cpp_sys`].
//!
//! This crate expects one or more `llama.cpp`-compatible GGUF models to be made available in
//! a directory specified in the `LLAMA_CPP_TEST_MODELS` environment variable.

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
    #[ignore]
    #[tokio::test]
    async fn load_models() {
        let dir = std::env::var("LLAMA_CPP_TEST_MODELS").unwrap_or_else(|_| {
            panic!(
                "LLAMA_CPP_TEST_MODELS environment variable not set. \
                Please set this to the directory containing one or more GGUF models."
            );
        });

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

        let dir = std::env::var("LLAMA_CPP_TEST_MODELS").unwrap_or_else(|_| {
            panic!(
                "LLAMA_CPP_TEST_MODELS environment variable not set. \
                Please set this to the directory containing one or more GGUF models."
            );
        });

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

        let dir = std::env::var("LLAMA_EMBED_MODELS_DIR").unwrap_or_else(|_| {
            panic!(
                "LLAMA_EMBED_MODELS_DIR environment variable not set. \
                Please set this to the directory containing one or more embedding GGUF models."
            );
        });

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
}