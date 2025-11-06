use anyhow::Result;
use gaba_train::auto_testing::AutoTestingSuite;

pub fn run_auto_tests() -> Result<()> {
    println!("Starting comprehensive auto-testing suite...\n");
    
    let mut suite = AutoTestingSuite::new();
    let _report = suite.run_all_tests()?;
    
    suite.print_report();
    
    println!("\nAuto-testing complete!");
    Ok(())
}

pub fn run_continuous_testing(interval_secs: u64) -> Result<()> {
    println!("Starting continuous auto-testing (interval: {}s)...\n", interval_secs);
    
    loop {
        let mut suite = AutoTestingSuite::new();
        let _report = suite.run_all_tests()?;
        suite.print_report();
        
        println!("\nWaiting {} seconds before next test cycle...", interval_secs);
        std::thread::sleep(std::time::Duration::from_secs(interval_secs));
    }
}
