//! Implements the [`Backend`] and [`BackendRef`] structs for managing llama.cpp
//! backends

use std::ptr;

use std::sync::Mutex;
use tracing::error;

use llama_cpp_sys::{
    ggml_backend_load_all, ggml_backend_load_all_from_path, ggml_numa_strategy, 
    llama_backend_free, llama_backend_init, llama_log_set, llama_numa_init,
};

use crate::detail;

/// The current instance of [`Backend`], if it exists. Also stored is a reference count used for
/// initialisation and freeing.
static BACKEND: Mutex<Option<(Backend, usize)>> = Mutex::new(None);

/// Empty struct used to initialise and free the [llama.cpp][llama.cpp] backend when it is created
/// dropped respectively.
///
/// [llama.cpp]: https://github.com/ggerganov/llama.cpp/
struct Backend {}

impl Backend {
    /// Initialises the [llama.cpp][llama.cpp] backend and sets its logger.
    ///
    /// There should only ever be one instance of this struct at any given time.
    ///
    /// [llama.cpp]: https://github.com/ggerganov/llama.cpp/
    fn init() -> Self {
        unsafe {
            // SAFETY: This is only called when no models or sessions exist.
            eprintln!("[Backend::init] Initializing llama backend...");
            llama_backend_init();
            
            // Load backends - check GGML_BACKENDS_PATH first (for dynamic loading)
            // This is the standard environment variable that llama.cpp uses
            let backend_path = if let Ok(ggml_path) = std::env::var("GGML_BACKENDS_PATH") {
                eprintln!("[Backend::init] Loading backends from GGML_BACKENDS_PATH: {}", ggml_path);
                Some(ggml_path)
            } else if let Some(build_path) = llama_cpp_sys::get_backends_build_path() {
                // Use the path from build time if available
                eprintln!("[Backend::init] Loading backends from build path: {}", build_path);
                Some(build_path.to_string())
            } else {
                eprintln!("[Backend::init] No backend path found, trying default locations...");
                None
            };
            
            if let Some(path) = backend_path {
                let c_path = std::ffi::CString::new(path.as_str()).unwrap();
                ggml_backend_load_all_from_path(c_path.as_ptr());
            } else {
                // Load all available backends from default locations
                eprintln!("[Backend::init] Loading all available backends from default locations...");
                ggml_backend_load_all();
            }
            
            // Check how many backends were loaded
            let backend_count = llama_cpp_sys::ggml_backend_reg_count();
            let device_count = llama_cpp_sys::ggml_backend_dev_count();
            eprintln!("[Backend::init] Backends registered: {}", backend_count);
            eprintln!("[Backend::init] Devices available: {}", device_count);

            // TODO look into numa strategies, this should probably be part of the API
            llama_numa_init(ggml_numa_strategy::GGML_NUMA_STRATEGY_DISTRIBUTE);

            // SAFETY: performs a simple assignment to static variables. Should only execute once
            // before any logs are made.
            llama_log_set(Some(detail::llama_log_callback), ptr::null_mut());
        }

        Self {}
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: This is only called when no models or sessions exist.
            llama_backend_free();
        }
    }
}

/// A "reference" to [`BACKEND`].
///
/// Initialises [`BACKEND`] if there is no [`Backend`] inside. If there are no other references,
/// this drops [`Backend`] upon getting itself dropped.
pub(crate) struct BackendRef {}

impl BackendRef {
    /// Creates a new reference, initialising [`BACKEND`] if necessary.
    pub(crate) fn new() -> Self {
        let mut lock = BACKEND.lock().unwrap();
        if let Some((_, count)) = lock.as_mut() {
            *count += 1;
        } else {
            let _ = lock.insert((Backend::init(), 1));
        }

        Self {}
    }
}

impl Drop for BackendRef {
    fn drop(&mut self) {
        let mut lock = BACKEND.lock().unwrap();
        if let Some((_, count)) = lock.as_mut() {
            *count -= 1;

            if *count == 0 {
                lock.take();
            }
        } else {
            error!("Backend as already been freed, this should never happen")
        }
    }
}

impl Clone for BackendRef {
    fn clone(&self) -> Self {
        Self::new()
    }
}
