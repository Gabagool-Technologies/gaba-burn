#[cfg(target_os = "linux")]
use std::fs;

#[derive(Debug, Clone, PartialEq)]
pub enum CpuVendor {
    Intel,
    AMD,
    Apple,
    ARM,
    RISCV,
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Platform {
    AppleSilicon,
    IntelXeon,
    AMDEPYC,
    ARMCortex,
    RISCV,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub vendor: CpuVendor,
    pub platform: Platform,
    pub num_cores: usize,
    pub l1_cache_size: usize,
    pub l2_cache_size: usize,
    pub l3_cache_size: usize,
    pub memory_bandwidth_gbps: f32,
}

impl HardwareInfo {
    pub fn detect() -> Self {
        #[cfg(target_os = "macos")]
        {
            Self::detect_macos()
        }
        #[cfg(target_os = "linux")]
        {
            Self::detect_linux()
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            Self::default()
        }
    }

    #[cfg(target_os = "macos")]
    fn detect_macos() -> Self {
        use std::process::Command;

        let brand = Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .unwrap_or_default();

        let cores = Command::new("sysctl")
            .args(["-n", "hw.ncpu"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(8);

        let l1_size = Command::new("sysctl")
            .args(["-n", "hw.l1dcachesize"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(65536);

        let l2_size = Command::new("sysctl")
            .args(["-n", "hw.l2cachesize"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(4194304);

        let l3_size = Command::new("sysctl")
            .args(["-n", "hw.l3cachesize"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);

        let (vendor, platform, bandwidth) = if brand.contains("Apple") {
            let bw = if brand.contains("M4") {
                273.0
            } else if brand.contains("M3") {
                200.0
            } else if brand.contains("M2") {
                100.0
            } else if brand.contains("M1") {
                68.0
            } else {
                100.0
            };
            (CpuVendor::Apple, Platform::AppleSilicon, bw)
        } else if brand.contains("Intel") {
            (CpuVendor::Intel, Platform::IntelXeon, 204.8)
        } else {
            (CpuVendor::Unknown, Platform::Unknown, 100.0)
        };

        Self {
            vendor,
            platform,
            num_cores: cores,
            l1_cache_size: l1_size,
            l2_cache_size: l2_size,
            l3_cache_size: l3_size,
            memory_bandwidth_gbps: bandwidth,
        }
    }

    #[cfg(target_os = "linux")]
    fn detect_linux() -> Self {
        let cpuinfo = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
        
        let vendor = if cpuinfo.contains("GenuineIntel") {
            CpuVendor::Intel
        } else if cpuinfo.contains("AuthenticAMD") {
            CpuVendor::AMD
        } else if cpuinfo.contains("ARM") {
            CpuVendor::ARM
        } else if cpuinfo.contains("riscv") {
            CpuVendor::RISCV
        } else {
            CpuVendor::Unknown
        };

        let cores = cpuinfo
            .lines()
            .filter(|l| l.starts_with("processor"))
            .count()
            .max(1);

        let l1_size = Self::read_cache_size("/sys/devices/system/cpu/cpu0/cache/index0/size")
            .unwrap_or(32768);
        let l2_size = Self::read_cache_size("/sys/devices/system/cpu/cpu0/cache/index2/size")
            .unwrap_or(262144);
        let l3_size = Self::read_cache_size("/sys/devices/system/cpu/cpu0/cache/index3/size")
            .unwrap_or(0);

        let (platform, bandwidth) = match vendor {
            CpuVendor::Intel => {
                if cpuinfo.contains("Xeon") {
                    (Platform::IntelXeon, 204.8)
                } else {
                    (Platform::Unknown, 100.0)
                }
            }
            CpuVendor::AMD => {
                if cpuinfo.contains("EPYC") {
                    (Platform::AMDEPYC, 204.8)
                } else {
                    (Platform::Unknown, 100.0)
                }
            }
            CpuVendor::ARM => (Platform::ARMCortex, 50.0),
            CpuVendor::RISCV => (Platform::RISCV, 50.0),
            _ => (Platform::Unknown, 100.0),
        };

        Self {
            vendor,
            platform,
            num_cores: cores,
            l1_cache_size: l1_size,
            l2_cache_size: l2_size,
            l3_cache_size: l3_size,
            memory_bandwidth_gbps: bandwidth,
        }
    }

    #[cfg(target_os = "linux")]
    fn read_cache_size(path: &str) -> Option<usize> {
        fs::read_to_string(path)
            .ok()
            .and_then(|s| {
                let s = s.trim();
                if s.ends_with('K') {
                    s[..s.len() - 1].parse::<usize>().ok().map(|v| v * 1024)
                } else if s.ends_with('M') {
                    s[..s.len() - 1].parse::<usize>().ok().map(|v| v * 1024 * 1024)
                } else {
                    s.parse().ok()
                }
            })
    }

    pub fn recommended_sequence(&self) -> &'static str {
        match self.platform {
            Platform::AppleSilicon => "extended",
            Platform::IntelXeon => "fibonacci",
            Platform::AMDEPYC => "extended",
            Platform::ARMCortex => "baseline",
            Platform::RISCV => "fibonacci",
            Platform::Unknown => "baseline",
        }
    }

    pub fn optimal_batch_size_hint(&self) -> usize {
        if self.num_cores >= 32 {
            67
        } else if self.num_cores >= 16 {
            42
        } else if self.num_cores >= 8 {
            26
        } else {
            13
        }
    }

    pub fn optimal_block_size_hint(&self) -> usize {
        if self.l3_cache_size > 100_000_000 {
            67
        } else if self.l3_cache_size > 20_000_000 {
            42
        } else if self.l2_cache_size > 2_000_000 {
            29
        } else if self.l2_cache_size > 500_000 {
            19
        } else {
            13
        }
    }
}

impl Default for HardwareInfo {
    fn default() -> Self {
        Self {
            vendor: CpuVendor::Unknown,
            platform: Platform::Unknown,
            num_cores: 8,
            l1_cache_size: 32768,
            l2_cache_size: 262144,
            l3_cache_size: 8388608,
            memory_bandwidth_gbps: 100.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_detection() {
        let info = HardwareInfo::detect();
        assert!(info.num_cores > 0);
        assert!(info.l1_cache_size >= 0);
    }

    #[test]
    fn test_recommended_sequence() {
        let apple = HardwareInfo {
            platform: Platform::AppleSilicon,
            ..Default::default()
        };
        assert_eq!(apple.recommended_sequence(), "extended");

        let intel = HardwareInfo {
            platform: Platform::IntelXeon,
            ..Default::default()
        };
        assert_eq!(intel.recommended_sequence(), "fibonacci");

        let amd = HardwareInfo {
            platform: Platform::AMDEPYC,
            ..Default::default()
        };
        assert_eq!(amd.recommended_sequence(), "extended");

        let arm = HardwareInfo {
            platform: Platform::ARMCortex,
            ..Default::default()
        };
        assert_eq!(arm.recommended_sequence(), "baseline");
    }

    #[test]
    fn test_optimal_hints() {
        let high_core = HardwareInfo {
            num_cores: 64,
            l3_cache_size: 256_000_000,
            ..Default::default()
        };
        assert_eq!(high_core.optimal_batch_size_hint(), 67);
        assert_eq!(high_core.optimal_block_size_hint(), 67);

        let low_core = HardwareInfo {
            num_cores: 4,
            l2_cache_size: 256_000,
            l3_cache_size: 0,
            ..Default::default()
        };
        assert_eq!(low_core.optimal_batch_size_hint(), 13);
        assert_eq!(low_core.optimal_block_size_hint(), 13);
    }
}
