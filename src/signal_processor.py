"""Signal processing implementation."""

import numpy as np
from scipy import signal as sig
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProcessingParams:
    """Signal processing parameters"""
    n: int = 2  # Number of cycles
    t1: float = 100.0  # Time for hyperpolarization (ms)
    t2: float = 100.0  # Time for depolarization (ms)
    V0: float = -80.0  # Baseline voltage (mV)
    V1: float = 100.0  # Hyperpolarization voltage (mV)
    V2: float = 10.0   # Depolarization voltage (mV)
    
    @property
    def delta_V1(self) -> float:
        """Calculate delta V1"""
        return self.V1 - self.V0
    
    @property
    def delta_V2(self) -> float:
        """Calculate delta V2"""
        return self.V2 - self.V0

class SignalProcessor:
    """Implements signal processing algorithms"""
    
    def __init__(self, params: ProcessingParams):
        self.params = params
        self.savgol_window = 21  # Window for Savitzky-Golay filter
        self.savgol_order = 3    # Polynomial order for filter
        
    def detect_baseline_end(self, signal: np.ndarray, time: np.ndarray) -> Tuple[int, float]:
        """Find end of baseline using linear regression error analysis."""
        window_size = 50  # Points to use for regression
        threshold_factor = 2.0  # How many std devs for error increase
        
        errors = []
        means = []
        
        # Sliding window linear regression
        for i in range(len(signal) - window_size):
            window = signal[i:i+window_size]
            t = time[i:i+window_size]
            
            # Fit line to window
            coeffs = np.polyfit(t, window, 1)
            fit = np.polyval(coeffs, t)
            
            # Calculate fit error
            error = np.mean((window - fit) ** 2)
            errors.append(error)
            means.append(np.mean(window))
            
        errors = np.array(errors)
        error_diff = np.diff(errors)
        
        # Find where error suddenly increases
        threshold = np.mean(error_diff) + threshold_factor * np.std(error_diff)
        change_points = np.where(error_diff > threshold)[0]
        
        if len(change_points) > 0:
            baseline_end = change_points[0]
            baseline_value = np.mean(means[:baseline_end])
            logger.info(f"Baseline ends at {time[baseline_end]:.2f} ms")
            return baseline_end, baseline_value
        else:
            logger.warning("No clear baseline end found, using first 10%")
            baseline_end = len(signal) // 10
            return baseline_end, np.mean(signal[:baseline_end])

    def denoise_signal(self, signal: np.ndarray) -> np.ndarray:
        """Advanced signal denoising combining multiple techniques"""
        # Remove spikes with median filter
        signal_medfilt = sig.medfilt(signal, kernel_size=5)
        
        # Smooth with Savitzky-Golay
        signal_smooth = sig.savgol_filter(
            signal_medfilt,
            window_length=self.savgol_window,
            polyorder=self.savgol_order
        )
        
        return signal_smooth
        
    def align_to_zero(self, signal: np.ndarray) -> np.ndarray:
        """Zero alignment using last 20 points"""
        zero_level = np.mean(signal[-20:])
        return signal - zero_level
    
    def detect_cycles(
        self,
        signal: np.ndarray,
        time: np.ndarray,
        baseline: float
    ) -> List[Dict[str, Any]]:
        """Detect hyperpolarization-depolarization cycles."""
        # More lenient thresholds for detection
        hyper_thresh = baseline - 0.1 * abs(self.params.delta_V1)  # Changed from 0.2 to 0.1
        depol_thresh = baseline + 0.1 * abs(self.params.delta_V2)  # Changed from 0.2 to 0.1
        min_cycle_duration = 0.5 * (self.params.t1 + self.params.t2)  # Changed from 0.8 to 0.5
        
        logger.info(f"Detection thresholds - Hyper: {hyper_thresh:.2f}, Depol: {depol_thresh:.2f}")
        logger.info(f"Signal range: {np.min(signal):.2f} to {np.max(signal):.2f}")
        
        cycles = []
        in_hyper = False
        
        for i in range(1, len(signal)):
            # Start of hyperpolarization 
            if not in_hyper and signal[i] < hyper_thresh:
                hyper_start = i
                in_hyper = True
                logger.debug(f"Found potential hyperpolarization at {time[i]:.2f} ms")
                
            # Start of depolarization
            elif in_hyper and signal[i] > depol_thresh:
                hyper_dur = time[i] - time[hyper_start]
                logger.debug(f"Found potential depolarization at {time[i]:.2f} ms")
                
                # More lenient timing validation
                if 0.5 * self.params.t1 <= hyper_dur <= 1.5 * self.params.t1:  # Wider timing window
                    depol_start = i
                    
                    # Find cycle end (return near baseline) with more tolerance
                    cycle_end = i
                    while cycle_end < len(signal) - 1 and \
                          abs(signal[cycle_end] - baseline) > 5.0:  # Changed from 2.0 to 5.0
                        cycle_end += 1
                    
                    cycle_dur = time[cycle_end] - time[hyper_start]
                    
                    # Store valid cycle
                    if cycle_dur >= min_cycle_duration:
                        cycle_info = {
                            'hyper_start': hyper_start,
                            'hyper_end': depol_start,
                            'depol_start': depol_start,
                            'depol_end': cycle_end,
                            'hyper_duration': hyper_dur,
                            'total_duration': cycle_dur,
                            'hyper_amplitude': baseline - np.min(signal[hyper_start:depol_start]),
                            'depol_amplitude': np.max(signal[depol_start:cycle_end]) - baseline
                        }
                        
                        cycles.append(cycle_info)
                        logger.info(f"Found valid cycle: Duration={cycle_dur:.2f}ms, "
                                  f"Hyper={cycle_info['hyper_amplitude']:.2f}mV, "
                                  f"Depol={cycle_info['depol_amplitude']:.2f}mV")
                        
                        if len(cycles) >= self.params.n:
                            break
                    else:
                        logger.debug(f"Cycle rejected: duration {cycle_dur:.2f}ms < {min_cycle_duration:.2f}ms")
                else:
                    logger.debug(f"Cycle rejected: hyper duration {hyper_dur:.2f}ms outside valid range")
                            
                in_hyper = False
        
        logger.info(f"Found {len(cycles)} valid cycles")
        return cycles

    def normalize_and_calculate_integral(
        self,
        signal: np.ndarray, 
        time: np.ndarray,
        cycles: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, float, List[float]]:
        """Normalize signal and calculate integrals"""
        # Normalize by delta_V1
        normalized = signal / abs(self.params.delta_V1)
        
        cycle_integrals = []
        
        for cycle in cycles:
            start = cycle['hyper_start']
            end = cycle['depol_end']
            
            # Calculate absolute integral for cycle
            cycle_signal = normalized[start:end]
            cycle_time = time[start:end] - time[start]
            integral = np.trapz(np.abs(cycle_signal), cycle_time)
            cycle_integrals.append(integral)
            
        total_integral = np.mean(cycle_integrals)
        return normalized, total_integral, cycle_integrals
    
    def verify_capacitance(self, integral_value: float) -> Tuple[bool, float]:
        """Verify if calculated capacitance matches expected ~1 µF/cm²"""
        delta_v = abs(self.params.V1 - self.params.V0) / 1000.0  # Convert to V
        capacitance = integral_value / delta_v  # µF/cm²
        
        # Check if within 20% of expected 1 µF/cm²
        is_valid = 0.8 <= capacitance <= 1.2
        
        if not is_valid:
            logger.warning(
                f"Calculated capacitance ({capacitance:.2f} µF/cm²) "
                "deviates significantly from expected 1 µF/cm²"
            )
        
        return is_valid, capacitance

    def process_signal(
        self,
        signal: np.ndarray,
        time: np.ndarray
    ) -> Dict[str, Any]:
        """Complete signal processing pipeline"""
        # Debug signal characteristics
        logger.info(f"Signal statistics:")
        logger.info(f"  Length: {len(signal)} points")
        logger.info(f"  Time range: {time[0]:.2f} to {time[-1]:.2f} ms")
        logger.info(f"  Value range: {np.min(signal):.2f} to {np.max(signal):.2f} mV")
        
        # Find baseline using regression analysis
        baseline_end, baseline = self.detect_baseline_end(signal, time)
        logger.info(f"Detected baseline: {baseline:.2f} mV")
        
        # Initial signal conditioning
        signal_baselined = signal - baseline
        logger.debug(f"Baselined signal range: {np.min(signal_baselined):.2f} to {np.max(signal_baselined):.2f} mV")
        
        # Denoise signal
        signal_denoised = self.denoise_signal(signal_baselined)
        logger.debug(f"Denoised signal range: {np.min(signal_denoised):.2f} to {np.max(signal_denoised):.2f} mV")
        
        # Align to zero using end points
        signal_zeroed = self.align_to_zero(signal_denoised)
        logger.debug(f"Zeroed signal range: {np.min(signal_zeroed):.2f} to {np.max(signal_zeroed):.2f} mV")
        
        # Plot debug visualization if needed
        if logger.getEffectiveLevel() <= logging.DEBUG:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 4))
            plt.plot(time, signal_zeroed, label='Processed Signal')
            plt.axhline(y=0, color='r', linestyle='--', label='Zero Line')
            plt.axvline(x=time[baseline_end], color='g', linestyle=':', label='Baseline End')
            plt.grid(True)
            plt.legend()
            plt.title('Signal After Processing')
            plt.xlabel('Time (ms)')
            plt.ylabel('Voltage (mV)')
            plt.savefig('debug_signal.png')
            plt.close()
        
        # Detect cycles (n cycles as specified)
        cycles = self.detect_cycles(signal_zeroed, time, 0.0)
        
        if not cycles:
            logger.error("Cycle detection failed. Signal characteristics:")
            logger.error(f"  Expected hyper threshold: {-0.1 * abs(self.params.delta_V1):.2f} mV")
            logger.error(f"  Expected depol threshold: {0.1 * abs(self.params.delta_V2):.2f} mV")
            logger.error(f"  Signal min: {np.min(signal_zeroed):.2f} mV")
            logger.error(f"  Signal max: {np.max(signal_zeroed):.2f} mV")
            raise ValueError("No valid cycles detected in signal")
        
        logger.info(f"Successfully detected {len(cycles)} cycles")
        for i, cycle in enumerate(cycles):
            logger.info(f"Cycle {i+1}:")
            logger.info(f"  Duration: {cycle['total_duration']:.2f} ms")
            logger.info(f"  Hyper amplitude: {cycle['hyper_amplitude']:.2f} mV")
            logger.info(f"  Depol amplitude: {cycle['depol_amplitude']:.2f} mV")
            
        # Normalize and calculate integrals
        normalized, integral, cycle_integrals = self.normalize_and_calculate_integral(
            signal_zeroed, time, cycles
        )
        
        logger.info(f"Calculated integral: {integral:.6f} V·s")
        logger.info(f"Individual cycle integrals: {[f'{x:.6f}' for x in cycle_integrals]}")
        
        # Verify capacitance
        is_valid, capacitance = self.verify_capacitance(integral)
        logger.info(f"Calculated capacitance: {capacitance:.2f} µF/cm²")
        logger.info(f"Capacitance verification: {'PASSED' if is_valid else 'FAILED'}")
        
        return {
            'processed_signal': signal_zeroed,
            'normalized_signal': normalized,
            'baseline': baseline,
            'baseline_end': baseline_end,
            'cycles': cycles,
            'integral': integral,
            'cycle_integrals': cycle_integrals,
            'capacitance': capacitance,
            'capacitance_valid': is_valid
        }