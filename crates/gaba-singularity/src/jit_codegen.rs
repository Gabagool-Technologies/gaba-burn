use crate::jit::{OptimizationLevel, WorkloadSignature};

pub struct X86CodeGenerator {
    optimization_level: OptimizationLevel,
}

impl X86CodeGenerator {
    pub fn new(opt_level: OptimizationLevel) -> Self {
        Self {
            optimization_level: opt_level,
        }
    }

    pub fn generate_gemm(&self, sig: &WorkloadSignature) -> Vec<u8> {
        match self.optimization_level {
            OptimizationLevel::O0 => self.generate_naive_gemm(sig),
            OptimizationLevel::O1 => self.generate_vectorized_gemm(sig),
            OptimizationLevel::O2 => self.generate_avx2_gemm(sig),
            OptimizationLevel::O3 => self.generate_avx512_gemm(sig),
        }
    }

    fn generate_naive_gemm(&self, _sig: &WorkloadSignature) -> Vec<u8> {
        let mut code = Vec::new();

        code.extend_from_slice(&[0x55, 0x48, 0x89, 0xe5]);

        code.extend_from_slice(&[0x48, 0x89, 0xf8, 0x48, 0x89, 0xf3, 0x48, 0x89, 0xd1]);

        code.extend_from_slice(&[0x5d, 0xc3]);

        code
    }

    fn generate_vectorized_gemm(&self, sig: &WorkloadSignature) -> Vec<u8> {
        let mut code = Vec::new();

        code.extend_from_slice(&[0x55, 0x48, 0x89, 0xe5]);

        if sig.m >= 4 && sig.n >= 4 && sig.k >= 4 {
            code.extend_from_slice(&[0x0f, 0x28, 0x07, 0x0f, 0x59, 0x06, 0x0f, 0x29, 0x01]);
        }

        code.extend_from_slice(&[0x5d, 0xc3]);

        code
    }

    fn generate_avx2_gemm(&self, sig: &WorkloadSignature) -> Vec<u8> {
        let mut code = Vec::new();

        code.extend_from_slice(&[0x55, 0x48, 0x89, 0xe5]);

        if sig.m >= 8 && sig.n >= 8 && sig.k >= 8 {
            code.extend_from_slice(&[
                0xc5, 0xfc, 0x28, 0x07, 0xc5, 0xfc, 0x59, 0x06, 0xc5, 0xfc, 0x29, 0x01,
            ]);
        }

        code.extend_from_slice(&[0x5d, 0xc3]);

        code
    }

    fn generate_avx512_gemm(&self, sig: &WorkloadSignature) -> Vec<u8> {
        let mut code = Vec::new();

        code.extend_from_slice(&[0x55, 0x48, 0x89, 0xe5]);

        if sig.m >= 16 && sig.n >= 16 && sig.k >= 16 {
            code.extend_from_slice(&[
                0x62, 0xf1, 0x7c, 0x48, 0x28, 0x07, 0x62, 0xf1, 0x7c, 0x48, 0x59, 0x06, 0x62, 0xf1,
                0x7c, 0x48, 0x29, 0x01,
            ]);
        }

        code.extend_from_slice(&[0x5d, 0xc3]);

        code
    }
}

pub fn generate_gemm_kernel(sig: &WorkloadSignature, opt_level: OptimizationLevel) -> Vec<u8> {
    let generator = X86CodeGenerator::new(opt_level);
    generator.generate_gemm(sig)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_registry::KernelType;

    #[test]
    fn test_generate_naive() {
        let sig = WorkloadSignature {
            m: 4,
            n: 4,
            k: 4,
            alignment: 64,
            kernel_type: KernelType::RustVectorized,
        };

        let code = generate_gemm_kernel(&sig, OptimizationLevel::O0);
        assert!(!code.is_empty());
        assert_eq!(code[0], 0x55);
        assert_eq!(code[code.len() - 1], 0xc3);
    }

    #[test]
    fn test_generate_vectorized() {
        let sig = WorkloadSignature {
            m: 8,
            n: 8,
            k: 8,
            alignment: 64,
            kernel_type: KernelType::RustVectorized,
        };

        let code = generate_gemm_kernel(&sig, OptimizationLevel::O1);
        assert!(!code.is_empty());
    }

    #[test]
    fn test_generate_avx2() {
        let sig = WorkloadSignature {
            m: 16,
            n: 16,
            k: 16,
            alignment: 64,
            kernel_type: KernelType::RustVectorized,
        };

        let code = generate_gemm_kernel(&sig, OptimizationLevel::O2);
        assert!(!code.is_empty());
    }
}
