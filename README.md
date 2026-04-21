# Predictive Maintenance System

A comprehensive Machine Learning-based Predictive Maintenance System for detecting machine abnormalities and scheduling maintenance activities.

## 📋 Overview

This system analyzes machine vibration signals to:
- Detect anomalies and potential failures
- Calculate automated Health Index scores
- Predict Remaining Useful Life (RUL)
- Recommend maintenance schedules

## ✨ Features

### Core Capabilities
- **Vibration Analysis**: FFT (Fast Fourier Transform) and Spectrogram analysis
- **Feature Extraction**: Time-domain and frequency-domain features
  - RMS, Peak, Crest Factor, Kurtosis, Skewness
  - Dominant frequency, Energy distribution
  - Band energy ratios
- **Health Index Calculation**: Automatic baseline fitting and health scoring (0-100)
- **RUL Prediction**: Exponential degradation modeling for remaining life estimation
- **Maintenance Scheduling**: Intelligent recommendations based on health status and failure modes

### Failure Mode Detection
- Bearing issues (high frequency energy)
- Rotor imbalance (high crest factor)
- Shaft misalignment (mid-low frequency energy)
- Gear wear

## 🏗️ Architecture

The system consists of five main components:

1. **VibrationAnalyzer**: Signal processing and feature extraction
2. **HealthIndexCalculator**: Baseline comparison and health scoring
3. **RULPredictor**: Remaining Useful Life prediction
4. **MaintenanceScheduler**: Maintenance recommendation engine
5. **PredictiveMaintenanceSystem**: Main orchestrator class

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Required packages: numpy, pandas, matplotlib, scipy

### Installation
```bash
pip install numpy pandas matplotlib scipy
```

### Basic Usage

```python
from predictive_maintenance_system import PredictiveMaintenanceSystem
import numpy as np

# Initialize the system
pms = PredictiveMaintenanceSystem(sampling_rate=1000)

# Train with normal vibration data
normal_data = [np.random.randn(1000) for _ in range(10)]
pms.train_baseline(normal_data)

# Analyze new vibration data
vibration_sample = np.random.randn(1000) * 1.5  # Simulated degraded state
results = pms.analyze(vibration_sample, timestamp=1)

# View results
print(f"Health Index: {results['health_index']:.2f}")
print(f"Status: {results['status']}")
print(f"Predicted RUL: {results['rul']} days")
print(f"Recommendations: {results['recommendations']}")

# Generate visualization
pms.plot_results([results])
plt.show()
```

## 📊 Health Index Scale

| Range | Status | Action |
|-------|--------|--------|
| 80-100 | Normal (ปกติ) | Routine monitoring |
| 60-80 | Warning (เฝ้าระวัง) | Plan inspection |
| 40-60 | Alert (อันตราย) | Schedule maintenance |
| 0-40 | Critical (วิกฤต) | Immediate action required |

## 🔧 API Reference

### PredictiveMaintenanceSystem

#### `__init__(sampling_rate=1000)`
Initialize the predictive maintenance system.

#### `train_baseline(normal_vibration_data)`
Train the system with normal operating condition data.
- **Parameters**: List of numpy arrays containing normal vibration signals

#### `analyze(vibration_data, timestamp=None)`
Analyze vibration data and return comprehensive results.
- **Returns**: Dictionary with features, health_index, rul, recommendations, and status

#### `plot_results(results_list)`
Generate visualization plots for health trends and predictions.

### VibrationAnalyzer

Extracts time-domain and frequency-domain features from vibration signals:
- `compute_fft()`: Fast Fourier Transform
- `compute_spectrogram()`: Time-frequency analysis
- `extract_features()`: Complete feature set extraction

### HealthIndexCalculator

Calculates equipment health based on deviation from baseline:
- `fit_baseline()`: Establish normal operating baseline
- `calculate_health_index()`: Compute current health score

### RULPredictor

Predicts remaining useful life:
- `fit_degradation_model()`: Learn degradation pattern
- `predict_rul()`: Estimate time to failure

### MaintenanceScheduler

Provides maintenance recommendations:
- `recommend_maintenance()`: Generate actionable maintenance plans

## 📈 Example Workflow

```python
# 1. Collect baseline data during normal operation
baseline_samples = collect_normal_vibration_data()

# 2. Train the system
pms = PredictiveMaintenanceSystem()
pms.train_baseline(baseline_samples)

# 3. Monitor equipment continuously
for i, sample in enumerate(monitoring_data):
    results = pms.analyze(sample, timestamp=i)
    
    if results['health_index'] < 60:
        print(f"⚠️ Warning: {results['status']}")
        for rec in results['recommendations']:
            print(f"  - {rec.get('message', rec.get('action', 'Check equipment'))}")

# 4. Visualize trends
pms.plot_results(all_results)
```

## 🎯 Use Cases

- Industrial machinery monitoring
- Rotating equipment maintenance
- Manufacturing line optimization
- Facility management
- IoT-enabled predictive maintenance

## 📝 License

This project is provided as-is for educational and commercial use.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📞 Support

For questions or issues, please open an issue in the repository.

---

**Note**: This system includes Thai language support for user-facing messages and maintains compatibility with international standards.
