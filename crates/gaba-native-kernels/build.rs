// build.rs is intentionally minimal. When the `zig` feature is enabled, this script
// could invoke the Zig compiler to produce a static library and emit cargo:rustc-link-lib
// and cargo:rustc-link-search directives. We keep it a no-op to avoid requiring Zig in CI.

fn main() {
    use std::env;
    use std::path::Path;
    use std::process::Command;

    use std::fs;
    println!("cargo:rerun-if-changed=src/lib.rs");

    // Collect all Zig sources under native/ and tell Cargo to rerun if any change.
    let mut zig_sources: Vec<String> = Vec::new();
    if let Ok(entries) = fs::read_dir("native") {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext == "zig" {
                    let s = path.to_string_lossy().to_string();
                    println!("cargo:rerun-if-changed={}", s);
                    zig_sources.push(s);
                }
            }
        }
    }

    // Cargo sets CARGO_FEATURE_<NAME>=1 when a feature is enabled.
    if env::var_os("CARGO_FEATURE_ZIG").is_none() {
        // Zig feature not enabled; nothing to build.
        return;
    }

    // Ensure zig is available
    let zig_check = Command::new("zig").arg("version").output();
    match zig_check {
        Ok(o) if o.status.success() => {
            // proceed
        }
        _ => panic!("feature `zig` enabled but `zig` was not found in PATH`"),
    }

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let target = env::var("TARGET").unwrap_or_else(|_| env::var("HOST").unwrap());

    // Choose platform library extension
    let lib_ext = if target.contains("darwin") {
        "dylib"
    } else if target.contains("windows") {
        "dll"
    } else {
        "so"
    };

    let out_path = Path::new(&out_dir).join(format!("libgemm.{}", lib_ext));

    // Create a root Zig file that exports all C functions from individual modules
    let combined = Path::new(&out_dir).join("combined_gemm.zig");
    {
        use std::fs::File;
        use std::io::Write;
        let mut f = File::create(&combined).expect("failed to create combined Zig source");

        // Write common imports and constants at the top
        writeln!(f, "const std = @import(\"std\");").ok();
        writeln!(f, "const math = std.math;").ok();
        writeln!(f, "").ok();
        writeln!(f, "// Common constants").ok();
        writeln!(f, "const SIMD_WIDTH: usize = 8;").ok();
        writeln!(f, "const BLOCK_M: usize = 64;").ok();
        writeln!(f, "const BLOCK_N: usize = 64;").ok();
        writeln!(f, "const BLOCK_K: usize = 128;").ok();
        writeln!(f, "const BLOCK_SIZE_M: usize = 64;").ok();
        writeln!(f, "const BLOCK_SIZE_N: usize = 64;").ok();
        writeln!(f, "const BLOCK_SIZE_K: usize = 256;").ok();
        writeln!(f, "const CACHE_LINE: usize = 128;").ok();
        writeln!(f, "").ok();

        // For each source file, inline its content but skip duplicate imports and constants
        for src in &zig_sources {
            let contents = fs::read_to_string(src).expect("failed to read zig source");
            writeln!(f, "// Begin {}\n", src).ok();

            // Filter out std imports and common constants
            for line in contents.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("const std = @import(\"std\")")
                    || trimmed.starts_with("const math = std.math")
                    || trimmed.starts_with("const SIMD_WIDTH: usize")
                    || trimmed.starts_with("const BLOCK_SIZE_M:")
                    || trimmed.starts_with("const BLOCK_SIZE_N:")
                    || trimmed.starts_with("const BLOCK_SIZE_K:")
                    || trimmed.starts_with("const BLOCK_M:")
                    || trimmed.starts_with("const BLOCK_N:")
                    || trimmed.starts_with("const BLOCK_K:")
                    || trimmed.starts_with("const CACHE_LINE:")
                {
                    // Skip duplicate imports and constants
                    continue;
                }
                writeln!(f, "{}", line).ok();
            }

            writeln!(f, "\n// End {}\n", src).ok();
        }
    }

    // Build the combined Zig source into a shared library.
    let status = Command::new("zig")
        .arg("cc")
        .arg("-OReleaseSmall")
        .arg("-fPIC")
        .arg("-shared")
        .arg(combined.as_os_str())
        .arg("-o")
        .arg(out_path.as_os_str())
        .status()
        .expect("failed to spawn `zig cc` to compile combined_gemm.zig");

    if !status.success() {
        panic!("`zig cc` failed to build combined_gemm.zig");
    }

    // Tell cargo to link the produced library and where to find it.
    println!("cargo:rustc-link-search=native={}", out_dir);
    // Link the dynamic library named `gemm` (libgemm.dylib / libgemm.so)
    println!("cargo:rustc-link-lib=dylib=gemm");
}
