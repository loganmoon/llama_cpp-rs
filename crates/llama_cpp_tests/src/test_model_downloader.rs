use std::fs;
use std::path::{Path, PathBuf};

/// Create minimal test GGUF files for testing
/// These are the absolute smallest valid GGUF files that can be loaded
pub struct TestModelGenerator {
    cache_dir: PathBuf,
}

impl TestModelGenerator {
    pub fn new() -> Self {
        let cache_dir = Self::get_cache_dir();
        fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");
        Self { cache_dir }
    }

    fn get_cache_dir() -> PathBuf {
        if let Ok(dir) = std::env::var("LLAMA_CPP_TEST_CACHE") {
            PathBuf::from(dir)
        } else {
            let mut path = std::env::temp_dir();
            path.push("llama_cpp_test_models");
            path
        }
    }

    /// Download a small test model from Hugging Face
    /// Using a very small model that's known to work
    pub async fn download_tiny_model(&self) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let model_name = "tinystories-15m-q4_0.gguf";
        let model_path = self.cache_dir.join(model_name);
        
        if model_path.exists() {
            println!("Test model already exists at {}", model_path.display());
            return Ok(model_path);
        }

        // Using TinyStories 15M model - one of the smallest valid models (~10MB)
        let url = "https://huggingface.co/TheBloke/TinyStories-15M-GGUF/resolve/main/tinystories-15m.Q4_0.gguf";
        
        println!("Downloading tiny test model (~10MB)...");
        let response = reqwest::get(url).await?;
        
        if !response.status().is_success() {
            return Err(format!("Failed to download: HTTP {}", response.status()).into());
        }

        let content = response.bytes().await?;
        fs::write(&model_path, content)?;
        
        println!("Test model downloaded to {}", model_path.display());
        Ok(model_path)
    }

    /// Download a small embedding model
    pub async fn download_tiny_embedding_model(&self) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let model_name = "all-minilm-l6-v2-q4_0.gguf";
        let model_path = self.cache_dir.join(model_name);
        
        if model_path.exists() {
            println!("Embedding model already exists at {}", model_path.display());
            return Ok(model_path);
        }

        // Using All-MiniLM-L6-v2 Q4_0 - small embedding model
        let url = "https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.Q4_0.gguf";
        
        println!("Downloading tiny embedding model (~23MB)...");
        let response = reqwest::get(url).await?;
        
        if !response.status().is_success() {
            return Err(format!("Failed to download: HTTP {}", response.status()).into());
        }

        let content = response.bytes().await?;
        fs::write(&model_path, content)?;
        
        println!("Embedding model downloaded to {}", model_path.display());
        Ok(model_path)
    }

    /// Setup test environment with minimal models
    pub async fn setup_test_environment(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("Setting up test environment with minimal models...");
        
        // Try to download small models
        match self.download_tiny_model().await {
            Ok(path) => {
                println!("Language model ready: {}", path.display());
            }
            Err(e) => {
                println!("Warning: Could not download language model: {}", e);
                println!("Tests requiring language models may fail");
            }
        }

        match self.download_tiny_embedding_model().await {
            Ok(path) => {
                println!("Embedding model ready: {}", path.display());
            }
            Err(e) => {
                println!("Warning: Could not download embedding model: {}", e);
                println!("Tests requiring embedding models may fail");
            }
        }

        // Set environment variables
        std::env::set_var("LLAMA_CPP_TEST_MODELS", self.cache_dir.to_str().unwrap());
        std::env::set_var("LLAMA_EMBED_MODELS_DIR", self.cache_dir.to_str().unwrap());
        
        println!("Test environment setup complete");
        println!("  Models directory: {}", self.cache_dir.display());
        
        Ok(())
    }

    pub fn get_models_dir(&self) -> PathBuf {
        self.cache_dir.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_setup_environment() {
        let generator = TestModelGenerator::new();
        generator.setup_test_environment().await.expect("Failed to setup test environment");
        
        assert!(std::env::var("LLAMA_CPP_TEST_MODELS").is_ok());
        assert!(std::env::var("LLAMA_EMBED_MODELS_DIR").is_ok());
    }
}