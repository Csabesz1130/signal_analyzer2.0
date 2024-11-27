"""Core signal processing module implementing professor's algorithms."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


@dataclass
class ProcessingParams:
    """Signal processing parameters as specified by professor"""

    n: int = 2  # Number of cycles to analyze
    t1: float = 100.0  # Time for hyperpolarization (ms)
    t2: float = 100.0  # Time for depolarization (ms)
    V0: float = -80.0  # Baseline voltage (mV)
    V1: float = 100.0  # Hyperpolarization voltage (mV)
    V2: float = 10.0  # Depolarization voltage (mV)

    @property
    def delta_V1(self) -> float:
        """Change in voltage for hyperpolarization"""
        return self.V1 - self.V0

    @property
    def delta_V2(self) -> float:
        """Change in voltage for depolarization"""
        return self.V2 - self.V0


class SignalProcessor:
    """Implements professor's signal processing algorithms"""

    def __init__(self, params: ProcessingParams):
        self.params = params
        self.savgol_window = 21  # Window for Savitzky-Golay filter
        self.savgol_order = 3  # Polynomial order for filter

    def detect_baseline_end(
        self, signal: np.ndarray, time: np.ndarray
    ) -> Tuple[int, float]:
        """
        Find end of baseline using linear regression error analysis.
        Professor's algorithm: Fit line to initial points, detect when error suddenly increases.

        Returns:
            Tuple[int, float]: (baseline end index, baseline voltage)
        """
        window_size = 50  # Points to use for regression
        threshold_factor = 2.0  # How many std devs for error increase

        errors = []
        means = []

        # Sliding window linear regression
        for i in range(len(signal) - window_size):
            window = signal[i : i + window_size]
            t = time[i : i + window_size]

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

    def align_to_zero(self, signal: np.ndarray) -> np.ndarray:
        """
        Professor's zero alignment: Use last 20 points to determine offset
        """
        zero_level = np.mean(signal[-20:])
        return signal - zero_level

    def denoise_signal(self, signal: np.ndarray) -> np.ndarray:
        """Advanced signal denoising combining multiple techniques"""
        # Remove spikes with median filter
        signal_medfilt = signal.medfilt(signal, kernel_size=5)

        # Smooth with Savitzky-Golay
        signal_smooth = signal.savgol_filter(
            signal_medfilt,
            window_length=self.savgol_window,
            polyorder=self.savgol_order,
        )

        return signal_smooth

    def detect_cycles(
        self, signal: np.ndarray, time: np.ndarray, baseline: float
    ) -> List[Dict[str, Any]]:
        """
        Detect hyperpolarization-depolarization cycles.
        Implements professor's approach looking for V1 and V2 transitions.
        """
        # Thresholds for detection
        hyper_thresh = baseline - 0.2 * abs(self.params.delta_V1)
        depol_thresh = baseline + 0.2 * abs(self.params.delta_V2)
        min_cycle_duration = 0.8 * (self.params.t1 + self.params.t2)

        cycles = []
        in_hyper = False
        last_trans = 0

        for i in range(1, len(signal)):
            # Start of hyperpolarization
            if not in_hyper and signal[i] < hyper_thresh:
                hyper_start = i
                in_hyper = True

            # Start of depolarization
            elif in_hyper and signal[i] > depol_thresh:
                hyper_dur = time[i] - time[hyper_start]

                # Validate timing
                if abs(hyper_dur - self.params.t1) <= 0.2 * self.params.t1:
                    depol_start = i

                    # Find cycle end (return near baseline)
                    cycle_end = i
                    while (
                        cycle_end < len(signal) - 1
                        and abs(signal[cycle_end] - baseline) > 2.0
                    ):
                        cycle_end += 1

                    cycle_dur = time[cycle_end] - time[hyper_start]

                    # Store valid cycle
                    if cycle_dur >= min_cycle_duration:
                        cycles.append(
                            {
                                "hyper_start": hyper_start,
                                "hyper_end": depol_start,
                                "depol_start": depol_start,
                                "depol_end": cycle_end,
                                "hyper_duration": hyper_dur,
                                "total_duration": cycle_dur,
                                "hyper_amplitude": baseline
                                - np.min(signal[hyper_start:depol_start]),
                                "depol_amplitude": np.max(signal[depol_start:cycle_end])
                                - baseline,
                            }
                        )

                        if len(cycles) >= self.params.n:
                            break

                in_hyper = False

        return cycles

    def normalize_and_calculate_integral(
        self, signal: np.ndarray, time: np.ndarray, cycles: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Normalize signal and calculate integrals for charge movement.
        Returns:
            - Normalized signal
            - Total integral
            - Per-cycle integrals
        """
        # Normalize by delta_V1
        normalized = signal / abs(self.params.delta_V1)

        cycle_integrals = []

        for cycle in cycles:
            start = cycle["hyper_start"]
            end = cycle["depol_end"]

            # Calculate absolute integral for cycle
            cycle_signal = normalized[start:end]
            cycle_time = time[start:end] - time[start]
            integral = np.trapz(np.abs(cycle_signal), cycle_time)
            cycle_integrals.append(integral)

        total_integral = np.mean(cycle_integrals)
        return normalized, total_integral, cycle_integrals

    def verify_capacitance(self, integral_value: float) -> Tuple[bool, float]:
        """
        Verify if calculated capacitance matches expected ~1 µF/cm²
        """
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

    def process_signal(self, signal: np.ndarray, time: np.ndarray) -> Dict[str, Any]:
        """
        Complete signal processing pipeline implementing professor's approach.
        """
        # 1. Find baseline using regression analysis
        baseline_end, baseline = self.detect_baseline_end(signal, time)
        logger.info(f"Detected baseline: {baseline:.2f} mV")

        # 2. Initial signal conditioning
        signal_baselined = signal - baseline
        signal_denoised = self.denoise_signal(signal_baselined)

        # 3. Align to zero using end points
        signal_zeroed = self.align_to_zero(signal_denoised)

        # 4. Detect cycles (n cycles as specified)
        cycles = self.detect_cycles(signal_zeroed, time, 0.0)

        if not cycles:
            raise ValueError("No valid cycles detected in signal")

        # 5. Normalize and calculate integrals
        normalized, integral, cycle_integrals = self.normalize_and_calculate_integral(
            signal_zeroed, time, cycles
        )

        # 6. Verify capacitance
        is_valid, capacitance = self.verify_capacitance(integral)

        return {
            "processed_signal": signal_zeroed,
            "normalized_signal": normalized,
            "baseline": baseline,
            "baseline_end": baseline_end,
            "cycles": cycles,
            "integral": integral,
            "cycle_integrals": cycle_integrals,
            "capacitance": capacitance,
            "capacitance_valid": is_valid,
        }
