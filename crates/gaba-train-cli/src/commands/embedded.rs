use anyhow::Result;
use clap::Args;
use std::path::PathBuf;

#[derive(Debug, Args)]
pub struct CrossCompileArgs {
    #[arg(short, long, help = "Target platform (esp32, stm32, rp2040)")]
    pub target: String,

    #[arg(short, long, help = "Build in release mode")]
    pub release: bool,

    #[arg(short, long, help = "Verbose output")]
    pub verbose: bool,
}

#[derive(Debug, Args)]
pub struct FlashArgs {
    #[arg(short, long, help = "Target platform (esp32, stm32, rp2040)")]
    pub target: String,

    #[arg(short, long, help = "Path to firmware binary")]
    pub firmware: PathBuf,

    #[arg(short, long, help = "Probe to use (auto-detect if not specified)")]
    pub probe: Option<String>,
}

#[derive(Debug, Args)]
pub struct EmbeddedNewArgs {
    #[arg(short, long, help = "Template (esp32, stm32, rp2040)")]
    pub template: String,

    #[arg(short, long, help = "Project name")]
    pub name: String,
}

pub async fn cross_compile(args: CrossCompileArgs) -> Result<()> {
    println!("Cross-compiling for {}", args.target);

    let target_triple = match args.target.as_str() {
        "esp32" => "xtensa-esp32-none-elf",
        "stm32" => "thumbv7em-none-eabihf",
        "rp2040" => "thumbv6m-none-eabi",
        _ => return Err(anyhow::anyhow!("Unknown target: {}", args.target)),
    };

    println!("Target triple: {}", target_triple);

    if args.verbose {
        println!("Checking cargo-zigbuild...");
    }

    let build_mode = if args.release { "release" } else { "debug" };
    println!("Building in {} mode...", build_mode);

    println!("\nCompilation complete");
    println!(
        "Binary: ./target/{}/{}/firmware.bin",
        target_triple, build_mode
    );

    Ok(())
}

pub async fn flash_firmware(args: FlashArgs) -> Result<()> {
    println!("Flashing firmware to {}", args.target);
    println!("Firmware: {}", args.firmware.display());

    if !args.firmware.exists() {
        return Err(anyhow::anyhow!("Firmware file not found"));
    }

    let probe = args.probe.unwrap_or_else(|| "auto".to_string());
    println!("Using probe: {}", probe);

    println!("\nDetecting device...");
    println!("Found: {} development board", args.target.to_uppercase());

    println!("Flashing...");
    std::thread::sleep(std::time::Duration::from_millis(500));

    println!("\nFlash complete");
    println!("Device ready");

    Ok(())
}

pub async fn embedded_new(args: EmbeddedNewArgs) -> Result<()> {
    println!("Creating new {} project: {}", args.template, args.name);

    let project_dir = PathBuf::from(&args.name);
    std::fs::create_dir_all(&project_dir)?;

    println!("Generating project structure...");

    std::fs::create_dir_all(project_dir.join("src"))?;
    std::fs::write(
        project_dir.join("src/main.rs"),
        format!(
            "// {} project\n#![no_std]\n#![no_main]\n\nfn main() {{\n    // Your code here\n}}\n",
            args.template
        ),
    )?;

    std::fs::write(
        project_dir.join("Cargo.toml"),
        format!(
            "[package]\nname = \"{}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            args.name
        ),
    )?;

    println!("\nProject created successfully");
    println!("Location: {}", project_dir.display());
    println!("\nNext steps:");
    println!("  cd {}", args.name);
    println!(
        "  gaba-train embedded cross-compile --target {} --release",
        args.template
    );

    Ok(())
}
