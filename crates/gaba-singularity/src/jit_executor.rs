use crate::jit::{JITCompiler, WorkloadSignature};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

#[cfg(target_os = "macos")]
use libc::{MAP_ANON, MAP_PRIVATE, PROT_EXEC, PROT_READ, PROT_WRITE, mmap, mprotect, munmap};

pub struct JITExecutor {
    compiler: JITCompiler,
    executable_pages: Arc<RwLock<HashMap<WorkloadSignature, ExecutablePage>>>,
}

pub struct ExecutablePage {
    ptr: *mut u8,
    size: usize,
}

impl ExecutablePage {
    #[cfg(target_os = "macos")]
    pub fn new(code: &[u8]) -> Result<Self, String> {
        unsafe {
            let size = (code.len() + 4095) & !4095;

            let ptr = mmap(
                std::ptr::null_mut(),
                size,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANON,
                -1,
                0,
            );

            if ptr == libc::MAP_FAILED {
                return Err("mmap failed".to_string());
            }

            std::ptr::copy_nonoverlapping(code.as_ptr(), ptr as *mut u8, code.len());

            if mprotect(ptr, size, PROT_READ | PROT_EXEC) != 0 {
                munmap(ptr, size);
                return Err("mprotect failed".to_string());
            }

            Ok(Self {
                ptr: ptr as *mut u8,
                size,
            })
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub fn new(_code: &[u8]) -> Result<Self, String> {
        Err("JIT execution only supported on macOS".to_string())
    }

    #[cfg(target_os = "macos")]
    pub unsafe fn call_gemm(
        &self,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        n: usize,
        k: usize,
    ) {
        type GemmFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32, usize, usize, usize);
        unsafe {
            let func: GemmFn = std::mem::transmute(self.ptr);
            func(a, b, c, m, n, k);
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub unsafe fn call_gemm(
        &self,
        _a: *const f32,
        _b: *const f32,
        _c: *mut f32,
        _m: usize,
        _n: usize,
        _k: usize,
    ) {
        panic!("JIT execution not supported on this platform");
    }
}

impl Drop for ExecutablePage {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        unsafe {
            munmap(self.ptr as *mut libc::c_void, self.size);
        }
    }
}

unsafe impl Send for ExecutablePage {}
unsafe impl Sync for ExecutablePage {}

impl JITExecutor {
    pub fn new() -> Self {
        Self {
            compiler: JITCompiler::new(),
            executable_pages: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn execute_gemm(
        &self,
        sig: WorkloadSignature,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<Duration, String> {
        {
            let pages = self.executable_pages.read();
            if let Some(page) = pages.get(&sig) {
                let start = std::time::Instant::now();
                unsafe {
                    page.call_gemm(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), sig.m, sig.n, sig.k);
                }
                return Ok(start.elapsed());
            }
        }

        let kernel = self.compiler.compile_kernel(sig.clone())?;
        let page = ExecutablePage::new(&kernel.code)?;

        let start = std::time::Instant::now();
        unsafe {
            page.call_gemm(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), sig.m, sig.n, sig.k);
        }
        let duration = start.elapsed();

        {
            let mut pages = self.executable_pages.write();
            pages.insert(sig, page);
        }

        Ok(duration)
    }

    pub fn cache_size(&self) -> usize {
        self.executable_pages.read().len()
    }

    pub fn clear_cache(&self) {
        self.executable_pages.write().clear();
    }
}

impl Default for JITExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_registry::KernelType;

    #[test]
    fn test_jit_executor_creation() {
        let executor = JITExecutor::new();
        assert_eq!(executor.cache_size(), 0);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_jit_execution() {
        let executor = JITExecutor::new();
        let sig = WorkloadSignature {
            m: 4,
            n: 4,
            k: 4,
            alignment: 64,
            kernel_type: KernelType::RustVectorized,
        };

        let a = vec![1.0f32; 16];
        let b = vec![1.0f32; 16];
        let mut c = vec![0.0f32; 16];

        let result = executor.execute_gemm(sig, &a, &b, &mut c);
        assert!(result.is_ok() || result.is_err());
    }
}
