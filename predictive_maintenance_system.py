#!/usr/bin/env python3
"""
Predictive Maintenance System - Machine Learning Based
ระบบตรวจจับความผิดปกติเครื่องจักรด้วย Machine Learning

Features:
- วิเคราะห์สัญญาณสั่นสะเทือน (FFT, Spectrogram)
- คำนวณ Health Index อัตโนมัติ
- ทำนาย RUL (Remaining Useful Life)
- แนะนำตารางซ่อมบำรุง
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, fft
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# Set Thai font for plots
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class VibrationAnalyzer:
    """วิเคราะห์สัญญาณสั่นสะเทือน"""
    
    def __init__(self, sampling_rate=1000):
        self.sampling_rate = sampling_rate
        
    def compute_fft(self, vibration_data):
        """คำนวณ Fast Fourier Transform"""
        n = len(vibration_data)
        frequencies = fft.fftfreq(n, 1/self.sampling_rate)
        fft_values = np.abs(fft.fft(vibration_data))
        
        # Return only positive frequencies
        positive_mask = frequencies >= 0
        return frequencies[positive_mask], fft_values[positive_mask]
    
    def compute_spectrogram(self, vibration_data, nperseg=256):
        """คำนวณ Spectrogram"""
        frequencies, times, Sxx = signal.spectrogram(
            vibration_data, 
            fs=self.sampling_rate, 
            nperseg=nperseg,
            noverlap=nperseg//2
        )
        return frequencies, times, Sxx
    
    def extract_features(self, vibration_data):
        """สกัดคุณลักษณะจากสัญญาณสั่นสะเทือน"""
        features = {}
        
        # Time domain features
        features['rms'] = np.sqrt(np.mean(vibration_data**2))
        features['peak'] = np.max(np.abs(vibration_data))
        features['crest_factor'] = features['peak'] / features['rms']
        features['kurtosis'] = pd.Series(vibration_data).kurtosis()
        features['skewness'] = pd.Series(vibration_data).skew()
        features['std'] = np.std(vibration_data)
        features['mean'] = np.mean(vibration_data)
        features['variance'] = np.var(vibration_data)
        
        # Frequency domain features
        freqs, fft_vals = self.compute_fft(vibration_data)
        features['dominant_freq'] = freqs[np.argmax(fft_vals)]
        features['freq_center'] = np.sum(freqs * fft_vals) / np.sum(fft_vals)
        features['freq_std'] = np.sqrt(np.sum(((freqs - features['freq_center'])**2) * fft_vals) / np.sum(fft_vals))
        
        # Energy features
        total_energy = np.sum(fft_vals**2)
        features['energy'] = total_energy
        
        # Band energy ratios
        n_bands = len(freqs) // 4
        if n_bands > 0:
            band_energies = []
            for i in range(4):
                start_idx = i * n_bands
                end_idx = (i + 1) * n_bands
                band_energy = np.sum(fft_vals[start_idx:end_idx]**2)
                band_energies.append(band_energy)
            
            features['low_freq_energy_ratio'] = band_energies[0] / total_energy if total_energy > 0 else 0
            features['mid_low_freq_energy_ratio'] = band_energies[1] / total_energy if total_energy > 0 else 0
            features['mid_high_freq_energy_ratio'] = band_energies[2] / total_energy if total_energy > 0 else 0
            features['high_freq_energy_ratio'] = band_energies[3] / total_energy if total_energy > 0 else 0
        
        return features


class HealthIndexCalculator:
    """คำนวณดัชนีสุขภาพเครื่องจักร"""
    
    def __init__(self):
        self.baseline_features = None
        self.feature_weights = None
        
    def fit_baseline(self, baseline_features_list):
        """กำหนดค่าพื้นฐานจากข้อมูลสถานะปกติ"""
        self.baseline_features = {}
        self.feature_weights = {}
        
        for key in baseline_features_list[0].keys():
            values = [f[key] for f in baseline_features_list]
            self.baseline_features[key] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            # Weight based on variability (lower std = higher weight)
            self.feature_weights[key] = 1.0 / (np.std(values) + 1e-6)
        
        # Normalize weights
        total_weight = sum(self.feature_weights.values())
        for key in self.feature_weights:
            self.feature_weights[key] /= total_weight
    
    def calculate_health_index(self, current_features):
        """คำนวณ Health Index (0-100, 100 = สุขภาพดีเยี่ยม)"""
        if self.baseline_features is None:
            raise ValueError("ต้อง fit baseline ก่อน")
        
        deviations = []
        for key in self.baseline_features:
            if key in current_features:
                baseline_mean = self.baseline_features[key]['mean']
                baseline_std = self.baseline_features[key]['std']
                current_value = current_features[key]
                
                # Calculate normalized deviation
                if baseline_std > 0:
                    deviation = abs(current_value - baseline_mean) / baseline_std
                else:
                    deviation = abs(current_value - baseline_mean)
                
                deviations.append((deviation, self.feature_weights[key]))
        
        # Weighted average deviation
        weighted_deviation = sum(d * w for d, w in deviations)
        
        # Convert to health index (exponential decay)
        health_index = 100 * np.exp(-weighted_deviation / 10)
        
        return max(0, min(100, health_index))


class RULPredictor:
    """ทำนาย Remaining Useful Life"""
    
    def __init__(self):
        self.degradation_model = None
        self.history = []
        
    def fit_degradation_model(self, health_indices, timestamps):
        """ฝึกโมเดลการเสื่อมสภาพ"""
        self.history = list(zip(timestamps, health_indices))
        
        # Fit exponential degradation model: HI(t) = 100 * exp(-lambda * t)
        if len(health_indices) > 1:
            # Linear regression on log scale
            valid_mask = np.array(health_indices) > 0
            if np.sum(valid_mask) > 1:
                log_hi = np.log(np.array(health_indices)[valid_mask])
                times = np.array(timestamps)[valid_mask]
                
                # Simple linear regression
                self.degradation_model = {
                    'slope': np.polyfit(times, log_hi, 1)[0],
                    'intercept': np.polyfit(times, log_hi, 1)[1]
                }
    
    def predict_rul(self, current_health_index, current_time, failure_threshold=20):
        """ทำนาย RUL จาก Health Index ปัจจุบัน"""
        if self.degradation_model is None:
            # Use simple linear extrapolation
            if len(self.history) < 2:
                return None
            
            recent_times = [h[0] for h in self.history[-10:]]
            recent_his = [h[1] for h in self.history[-10:]]
            
            if len(recent_times) > 1:
                slope = (recent_his[-1] - recent_his[0]) / (recent_times[-1] - recent_times[0])
                if slope >= 0:
                    return float('inf')
                
                time_to_failure = (failure_threshold - current_health_index) / slope
                return max(0, time_to_failure)
            return None
        
        # Use exponential model
        lambda_param = -self.degradation_model['slope']
        if lambda_param <= 0:
            return float('inf')
        
        # Time when HI will reach failure threshold
        log_current = np.log(max(current_health_index, 1e-6))
        log_threshold = np.log(max(failure_threshold, 1e-6))
        
        time_to_failure = (log_threshold - log_current) / lambda_param
        return max(0, time_to_failure)


class MaintenanceScheduler:
    """แนะนำตารางซ่อมบำรุง"""
    
    def __init__(self):
        self.maintenance_history = []
        self.failure_modes = {
            'bearing': {'threshold': 40, 'lead_time': 7},
            'imbalance': {'threshold': 50, 'lead_time': 3},
            'misalignment': {'threshold': 45, 'lead_time': 5},
            'gear_wear': {'threshold': 35, 'lead_time': 10}
        }
    
    def recommend_maintenance(self, health_index, rul, features=None):
        """แนะนำตารางซ่อมบำรุง"""
        recommendations = []
        
        # Urgency level
        if health_index < 20:
            urgency = "ฉุกเฉิน (Emergency)"
            priority = 1
        elif health_index < 40:
            urgency = "เร่งด่วน (Urgent)"
            priority = 2
        elif health_index < 60:
            urgency = "ควรดำเนินการเร็วๆ นี้ (Soon)"
            priority = 3
        elif health_index < 80:
            urgency = "วางแผน (Planned)"
            priority = 4
        else:
            urgency = "ปกติ (Normal)"
            priority = 5
        
        recommendations.append({
            'type': 'overall_health',
            'urgency': urgency,
            'priority': priority,
            'health_index': health_index,
            'rul_days': rul
        })
        
        # Specific maintenance actions based on features
        if features:
            # Check for bearing issues (high frequency energy)
            if features.get('high_freq_energy_ratio', 0) > 0.3:
                recommendations.append({
                    'type': 'bearing_inspection',
                    'urgency': 'สูง' if health_index < 40 else 'ปานกลาง',
                    'action': 'ตรวจสอบและเปลี่ยนตลับลูกปืน',
                    'lead_time_days': self.failure_modes['bearing']['lead_time']
                })
            
            # Check for imbalance (high crest factor, dominant low freq)
            if features.get('crest_factor', 0) > 3.0:
                recommendations.append({
                    'type': 'balancing',
                    'urgency': 'สูง' if health_index < 50 else 'ปานกลาง',
                    'action': 'ปรับสมดุลโรเตอร์',
                    'lead_time_days': self.failure_modes['imbalance']['lead_time']
                })
            
            # Check for misalignment
            if features.get('mid_low_freq_energy_ratio', 0) > 0.4:
                recommendations.append({
                    'type': 'alignment',
                    'urgency': 'สูง' if health_index < 45 else 'ปานกลาง',
                    'action': 'จัดแนวเพลาใหม่',
                    'lead_time_days': self.failure_modes['misalignment']['lead_time']
                })
        
        # Schedule recommendation
        if rul is not None and rul != float('inf'):
            recommended_schedule = max(1, rul * 0.7)  # Schedule at 70% of RUL
            recommendations.append({
                'type': 'schedule',
                'recommended_days': recommended_schedule,
                'latest_days': rul * 0.9,
                'message': f"ควรจัดตารางซ่อมบำรุงในอีก {recommended_schedule:.1f} วัน (ไม่เกิน {rul * 0.9:.1f} วัน)"
            })
        
        return recommendations


class PredictiveMaintenanceSystem:
    """ระบบ Predictive Maintenance แบบครบวงจร"""
    
    def __init__(self, sampling_rate=1000):
        self.vibration_analyzer = VibrationAnalyzer(sampling_rate)
        self.health_calculator = HealthIndexCalculator()
        self.rul_predictor = RULPredictor()
        self.maintenance_scheduler = MaintenanceScheduler()
        self.data_history = []
        self.is_trained = False
    
    def train_baseline(self, normal_vibration_data):
        """ฝึกโมเดลด้วยข้อมูลปกติ"""
        print("📊 กำลังฝึกโมเดลด้วยข้อมูลปกติ...")
        
        baseline_features = []
        for data in normal_vibration_data:
            features = self.vibration_analyzer.extract_features(data)
            baseline_features.append(features)
        
        self.health_calculator.fit_baseline(baseline_features)
        self.is_trained = True
        print("✅ การฝึกโมเดลเสร็จสิ้น")
    
    def analyze(self, vibration_data, timestamp=None):
        """วิเคราะห์ข้อมูลและให้คำแนะนำ"""
        if not self.is_trained:
            raise ValueError("ต้อง train baseline ก่อนใช้งาน")
        
        # Extract features
        features = self.vibration_analyzer.extract_features(vibration_data)
        
        # Calculate health index
        health_index = self.health_calculator.calculate_health_index(features)
        
        # Update history
        if timestamp is None:
            timestamp = len(self.data_history)
        
        self.data_history.append({
            'timestamp': timestamp,
            'features': features,
            'health_index': health_index
        })
        
        # Update RUL predictor
        timestamps = [d['timestamp'] for d in self.data_history]
        health_indices = [d['health_index'] for d in self.data_history]
        self.rul_predictor.fit_degradation_model(health_indices, timestamps)
        
        # Predict RUL
        rul = self.rul_predictor.predict_rul(health_index, timestamp)
        
        # Get maintenance recommendations
        recommendations = self.maintenance_scheduler.recommend_maintenance(
            health_index, rul, features
        )
        
        return {
            'timestamp': timestamp,
            'features': features,
            'health_index': health_index,
            'rul': rul,
            'recommendations': recommendations,
            'status': self._get_status(health_index)
        }
    
    def _get_status(self, health_index):
        """ระบุสถานะจาก Health Index"""
        if health_index >= 80:
            return "ปกติ (Normal)"
        elif health_index >= 60:
            return "เฝ้าระวัง (Warning)"
        elif health_index >= 40:
            return "อันตราย (Alert)"
        else:
            return "วิกฤต (Critical)"
    
    def plot_results(self, results_list):
        """แสดงผลกราฟ"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Health Index Trend
        timestamps = [r['timestamp'] for r in results_list]
        health_indices = [r['health_index'] for r in results_list]
        
        axes[0, 0].plot(timestamps, health_indices, 'b-o', linewidth=2, markersize=6)
        axes[0, 0].axhline(y=80, color='g', linestyle='--', label='ดี (>80)')
        axes[0, 0].axhline(y=60, color='y', linestyle='--', label='เฝ้าระวัง (60-80)')
        axes[0, 0].axhline(y=40, color='orange', linestyle='--', label='อันตราย (40-60)')
        axes[0, 0].axhline(y=20, color='r', linestyle='--', label='วิกฤต (<40)')
        axes[0, 0].set_xlabel('เวลา (Time)')
        axes[0, 0].set_ylabel('Health Index')
        axes[0, 0].set_title('แนวโน้ม Health Index')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: RUL Prediction
        ruls = [r['rul'] if r['rul'] != float('inf') else None for r in results_list]
        valid_ruls = [(t, r) for t, r in zip(timestamps, ruls) if r is not None]
        if valid_ruls:
            valid_times, valid_ruls = zip(*valid_ruls)
            axes[0, 1].plot(valid_times, valid_ruls, 'r-o', linewidth=2, markersize=6)
            axes[0, 1].set_xlabel('เวลา (Time)')
            axes[0, 1].set_ylabel('RUL (วัน)')
            axes[0, 1].set_title('การทำนาย Remaining Useful Life')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Feature Evolution (Crest Factor)
        crest_factors = [r['features']['crest_factor'] for r in results_list]
        axes[1, 0].plot(timestamps, crest_factors, 'g-o', linewidth=2, markersize=6)
        axes[1, 0].axhline(y=3.0, color='r', linestyle='--', label='Threshold')
        axes[1, 0].set_xlabel('เวลา (Time)')
        axes[1, 0].set_ylabel('Crest Factor')
        axes[1, 0].set_title('แนวโน้ม Crest Factor')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Status Distribution
        status_counts = {}
        for r in results_list:
            status = r['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        colors = {'ปกติ (Normal)': 'green', 'เฝ้าระวัง (Warning)': 'yellow', 
                 'อันตราย (Alert)': 'orange', 'วิกฤต (Critical)': 'red'}
        statuses = list(status_counts.keys())
        counts = [status_counts[s] for s in statuses]
        bar_colors = [colors.get(s, 'blue') for s in statuses]
        
        axes[1, 1].bar(statuses, counts, color=bar_colors, edgecolor='black')
        axes[1, 1].set_xlabel('สถานะ')
        axes[1, 1].set_ylabel('จำนวนครั้ง')
        axes[1, 1].set_title('การกระจายสถานะ')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('/workspace/predictive_maintenance_analysis.png', dpi=150, bbox_inches='tight')
        print("\n📈 กราฟผลลัพธ์ถูกบันทึกที่: /workspace/predictive_maintenance_analysis.png")
        plt.close()


def generate_synthetic_data():
    """สร้างชุดข้อมูลตัวอย่างสำหรับทดสอบ"""
    print("=" * 70)
    print("🔧 ระบบ Predictive Maintenance - เครื่องกำเนิดข้อมูลทดสอบ")
    print("=" * 70)
    
    np.random.seed(42)
    sampling_rate = 1000  # Hz
    duration = 1  # วินาที
    n_samples = sampling_rate * duration
    
    # Generate normal operation data (first 50 samples)
    print("\n📝 กำลังสร้างข้อมูลสถานะปกติ (50 ตัวอย่าง)...")
    normal_data = []
    for i in range(50):
        # Normal vibration: low amplitude, random noise
        t = np.linspace(0, duration, n_samples)
        base_freq = 50  # Hz
        vibration = (
            0.5 * np.sin(2 * np.pi * base_freq * t) +
            0.2 * np.sin(2 * np.pi * 2 * base_freq * t) +
            0.1 * np.random.randn(n_samples)
        )
        normal_data.append(vibration)
    
    # Generate degrading data (next 50 samples showing progressive degradation)
    print("📝 กำลังสร้างข้อมูลเสื่อมสภาพ (50 ตัวอย่าง)...")
    degrading_data = []
    for i in range(50):
        t = np.linspace(0, duration, n_samples)
        base_freq = 50
        
        # Progressive degradation
        degradation_factor = 1 + (i / 50) * 3
        
        # Add fault signatures
        bearing_fault_freq = 150  # Hz
        imbalance_freq = 25  # Hz
        
        vibration = (
            0.5 * degradation_factor * np.sin(2 * np.pi * base_freq * t) +
            0.3 * degradation_factor * np.sin(2 * np.pi * imbalance_freq * t) +
            0.4 * degradation_factor * np.sin(2 * np.pi * bearing_fault_freq * t) +
            0.2 * degradation_factor * np.random.randn(n_samples)
        )
        
        # Add impacts (simulating bearing defects)
        if i > 20:
            impact_rate = int(10 + (i - 20) * 2)
            impact_indices = np.random.choice(n_samples, impact_rate, replace=False)
            vibration[impact_indices] += 2 * degradation_factor * np.sign(np.random.randn(impact_rate))
        
        degrading_data.append(vibration)
    
    # Combine all data
    all_data = normal_data + degrading_data
    
    # Create DataFrame with metadata
    timestamps = list(range(len(all_data)))
    labels = ['normal'] * 50 + ['degrading'] * 50
    
    print(f"\n✅ สร้างข้อมูลสำเร็จ: {len(all_data)} ตัวอย่าง")
    print(f"   - สถานะปกติ: 50 ตัวอย่าง")
    print(f"   - เสื่อมสภาพ: 50 ตัวอย่าง")
    
    return all_data, timestamps, labels, sampling_rate


def main():
    """ฟังก์ชันหลัก"""
    print("\n" + "=" * 70)
    print("🎯 ระบบ Predictive Maintenance ด้วย Machine Learning")
    print("   ตรวจจับความผิดปกติเครื่องจักร - ทำนาย RUL - แนะนำการซ่อมบำรุง")
    print("=" * 70)
    
    # Generate synthetic data
    all_data, timestamps, labels, sampling_rate = generate_synthetic_data()
    
    # Initialize system
    print("\n🚀 กำลังเริ่มต้นระบบ...")
    pm_system = PredictiveMaintenanceSystem(sampling_rate=sampling_rate)
    
    # Train with normal data
    normal_data = all_data[:50]
    pm_system.train_baseline(normal_data)
    
    # Analyze all data
    print("\n🔍 กำลังวิเคราะห์ข้อมูลทั้งหมด...")
    results = []
    
    for i, (data, ts, label) in enumerate(zip(all_data, timestamps, labels)):
        result = pm_system.analyze(data, timestamp=ts)
        results.append(result)
        
        # Print progress every 10 samples
        if (i + 1) % 10 == 0:
            print(f"   วิเคราะห์แล้ว {i + 1}/{len(all_data)} ตัวอย่าง")
    
    # Display summary
    print("\n" + "=" * 70)
    print("📋 สรุปผลการวิเคราะห์")
    print("=" * 70)
    
    # Overall statistics
    final_result = results[-1]
    print(f"\n📊 Health Index สุดท้าย: {final_result['health_index']:.2f}")
    print(f"📊 สถานะปัจจุบัน: {final_result['status']}")
    
    if final_result['rul'] is not None and final_result['rul'] != float('inf'):
        print(f"⏰ RUL ที่ทำนาย: {final_result['rul']:.1f} วัน")
    else:
        print("⏰ RUL: ยังไม่สามารถประเมินได้")
    
    # Show maintenance recommendations
    print("\n🔧 คำแนะนำการซ่อมบำรุง:")
    for rec in final_result['recommendations']:
        if rec['type'] == 'overall_health':
            print(f"   • ความเร่งด่วน: {rec['urgency']}")
            print(f"   • Health Index: {rec['health_index']:.2f}")
        elif rec['type'] == 'schedule':
            print(f"   • {rec['message']}")
        else:
            print(f"   • {rec.get('action', rec['type'])}: {rec.get('urgency', '')}")
    
    # Detailed analysis for specific time points
    print("\n" + "=" * 70)
    print("📈 การวิเคราะห์แบบละเอียด (บางจุด)")
    print("=" * 70)
    
    sample_points = [0, 25, 50, 75, 99]
    for idx in sample_points:
        r = results[idx]
        print(f"\n⏱️  เวลาที่ {r['timestamp']}:")
        print(f"   Health Index: {r['health_index']:.2f} ({r['status']})")
        print(f"   Crest Factor: {r['features']['crest_factor']:.3f}")
        print(f"   RMS: {r['features']['rms']:.4f}")
        print(f"   Dominant Freq: {r['features']['dominant_freq']:.1f} Hz")
        if r['rul'] is not None and r['rul'] != float('inf'):
            print(f"   RUL: {r['rul']:.1f} วัน")
    
    # Plot results
    print("\n📊 กำลังสร้างกราฟแสดงผล...")
    pm_system.plot_results(results)
    
    # Save detailed results to CSV
    print("\n💾 กำลังบันทึกรายละเอียดผลวิเคราะห์...")
    
    results_df = pd.DataFrame([
        {
            'timestamp': r['timestamp'],
            'health_index': r['health_index'],
            'status': r['status'],
            'rul': r['rul'] if r['rul'] != float('inf') else None,
            'rms': r['features']['rms'],
            'peak': r['features']['peak'],
            'crest_factor': r['features']['crest_factor'],
            'kurtosis': r['features']['kurtosis'],
            'dominant_freq': r['features']['dominant_freq'],
            'energy': r['features']['energy']
        }
        for r in results
    ])
    
    results_df.to_csv('/workspace/maintenance_analysis_results.csv', index=False)
    print("✅ ผลลัพธ์ถูกบันทึกที่: /workspace/maintenance_analysis_results.csv")
    
    # Save sample raw data
    sample_data_df = pd.DataFrame({
        'sample_id': list(range(len(all_data))),
        'label': labels,
        'data_length': [len(d) for d in all_data]
    })
    sample_data_df.to_csv('/workspace/sample_metadata.csv', index=False)
    
    # Save first few samples as numpy arrays (use all_data instead of separate lists)
    np.save('/workspace/sample_vibration_normal.npy', np.array(all_data[:5]))
    np.save('/workspace/sample_vibration_degrading.npy', np.array(all_data[50:55]))
    print("✅ ข้อมูลตัวอย่างถูกบันทึกที่: /workspace/sample_vibration_*.npy")
    
    print("\n" + "=" * 70)
    print("✅ การวิเคราะห์เสร็จสิ้น!")
    print("=" * 70)
    print("\n📁 ไฟล์ที่สร้างขึ้น:")
    print("   1. predictive_maintenance_analysis.png - กราฟผลการวิเคราะห์")
    print("   2. maintenance_analysis_results.csv - ผลวิเคราะห์แบบละเอียด")
    print("   3. sample_metadata.csv - เมตาดาต้าของข้อมูลตัวอย่าง")
    print("   4. sample_vibration_normal.npy - ข้อมูลสั่นสะเทือนปกติ")
    print("   5. sample_vibration_degrading.npy - ข้อมูลสั่นสะเทือนเสื่อมสภาพ")
    print("\n🎯 ระบบพร้อมใช้งาน!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
