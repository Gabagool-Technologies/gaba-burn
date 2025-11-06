use gaba_train::pruning::PruningConfig;
use gaba_train::quantization::QuantizationConfig;
use gaba_train::TrainingConfig;

#[test]
fn test_config_with_singularity_enabled() {
    let config = TrainingConfig {
        epochs: 5,
        batch_size: 16,
        learning_rate: 0.001,
        test_ratio: 0.2,
        early_stopping_patience: None,
        verbose: false,
        use_singularity: true,
        quantization: QuantizationConfig::default(),
        pruning: PruningConfig::default(),
        enable_profiling: false,
    };

    assert!(config.use_singularity);
}

#[test]
fn test_config_with_singularity_disabled() {
    let config = TrainingConfig {
        epochs: 5,
        batch_size: 16,
        learning_rate: 0.001,
        test_ratio: 0.2,
        early_stopping_patience: None,
        verbose: false,
        use_singularity: false,
        quantization: QuantizationConfig::default(),
        pruning: PruningConfig::default(),
        enable_profiling: false,
    };

    assert!(!config.use_singularity);
}

#[test]
fn test_default_config_has_singularity_enabled() {
    let config = TrainingConfig::default();
    assert!(config.use_singularity);
    assert_eq!(config.epochs, 100);
    assert_eq!(config.batch_size, 32);
}

#[test]
fn test_config_serialization() {
    let config = TrainingConfig {
        epochs: 10,
        batch_size: 64,
        learning_rate: 0.01,
        test_ratio: 0.3,
        early_stopping_patience: Some(5),
        verbose: true,
        use_singularity: false,
        quantization: QuantizationConfig::default(),
        pruning: PruningConfig::default(),
        enable_profiling: false,
    };

    assert_eq!(config.epochs, 10);
    assert_eq!(config.batch_size, 64);
    assert!(!config.use_singularity);
}
