"""Visualization module for signal analysis results."""

import logging
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class SignalVisualizer:
    """Creates comprehensive visualizations of signal analysis"""

    def __init__(self):
        self.colors = {
            "raw": "#1f77b4",  # Blue
            "processed": "#2ca02c",  # Green
            "baseline": "#ff7f0e",  # Orange
            "cycles": "#d62728",  # Red
            "mirror": "#9467bd",  # Purple
        }

    def plot_signal_processing(
        self,
        ax: plt.Axes,
        time: np.ndarray,
        raw_signal: np.ndarray,
        processed_signal: np.ndarray,
        baseline: float,
        baseline_end: int,
    ) -> None:
        """Plot raw and processed signals with baseline"""
        # Raw signal
        ax.plot(
            time, raw_signal, color=self.colors["raw"], label="Raw Signal", alpha=0.5
        )

        # Processed signal
        ax.plot(
            time,
            processed_signal,
            color=self.colors["processed"],
            label="Processed Signal",
        )

        # Baseline indicator
        ax.axhline(
            y=baseline, color=self.colors["baseline"], linestyle="--", label="Baseline"
        )
        ax.axvline(
            x=time[baseline_end],
            color=self.colors["baseline"],
            linestyle=":",
            alpha=0.5,
        )

        ax.set_title("Signal Processing Steps")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Voltage (mV)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    def plot_cycle_detection(
        self,
        ax: plt.Axes,
        time: np.ndarray,
        signal: np.ndarray,
        cycles: List[Dict[str, Any]],
    ) -> None:
        """Plot detected hyperpolarization-depolarization cycles"""
        # Plot signal
        ax.plot(time, signal, color=self.colors["processed"])

        # Highlight cycles
        for i, cycle in enumerate(cycles):
            # Hyperpolarization phase
            ax.axvspan(
                time[cycle["hyper_start"]],
                time[cycle["hyper_end"]],
                color=self.colors["cycles"],
                alpha=0.1,
                label=f"Cycle {i+1} Hyper" if i == 0 else "",
            )

            # Depolarization phase
            ax.axvspan(
                time[cycle["depol_start"]],
                time[cycle["depol_end"]],
                color=self.colors["cycles"],
                alpha=0.2,
                label=f"Cycle {i+1} Depol" if i == 0 else "",
            )

            # Annotate cycle info
            mid_time = time[cycle["hyper_start"]] + cycle["total_duration"] / 2
            ax.text(
                mid_time,
                np.max(signal),
                f"Cycle {i+1}\n"
                f"Hyper: {cycle['hyper_duration']:.1f}ms\n"
                f"Total: {cycle['total_duration']:.1f}ms",
                horizontalalignment="center",
                verticalalignment="bottom",
            )

        ax.set_title("Detected Cycles")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Voltage (mV)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    def plot_normalized_integral(
        self,
        ax: plt.Axes,
        time: np.ndarray,
        normalized: np.ndarray,
        cycles: List[Dict[str, Any]],
        integral: float,
        capacitance: float,
    ) -> None:
        """Plot normalized signal with integral visualization"""
        # Plot normalized signal
        ax.plot(time, normalized, color=self.colors["processed"], label="Normalized")

        # Plot mirrored signal for integral visualization
        ax.plot(
            time,
            -np.abs(normalized),
            color=self.colors["mirror"],
            label="Mirrored",
            alpha=0.5,
        )

        # Fill integral areas
        for cycle in cycles:
            start = cycle["hyper_start"]
            end = cycle["depol_end"]
            cycle_time = time[start:end]
            cycle_signal = normalized[start:end]

            # Fill between signal and its mirror
            ax.fill_between(
                cycle_time,
                cycle_signal,
                -np.abs(cycle_signal),
                color=self.colors["cycles"],
                alpha=0.2,
            )

        # Add integral and capacitance info
        ax.text(
            0.02,
            0.98,
            f"Integral: {integral:.6f} V·s\n" f"Capacitance: {capacitance:.2f} µF/cm²",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.7),
        )

        ax.set_title("Normalized Signal and Integral")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Normalized Amplitude")
        ax.grid(True, alpha=0.3)
        ax.legend()

    def create_analysis_figure(
        self, time: np.ndarray, raw_signal: np.ndarray, results: Dict[str, Any]
    ) -> plt.Figure:
        """Create complete analysis figure with all plots"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        plt.subplots_adjust(hspace=0.4)

        # Plot original signal processing
        self.plot_signal_processing(
            axes[0],
            time,
            raw_signal,
            results["processed_signal"],
            results["baseline"],
            results["baseline_end"],
        )

        # Plot cycle detection
        self.plot_cycle_detection(
            axes[1], time, results["processed_signal"], results["cycles"]
        )

        # Plot normalized signal and integral
        self.plot_normalized_integral(
            axes[2],
            time,
            results["normalized_signal"],
            results["cycles"],
            results["integral"],
            results["capacitance"],
        )

        return fig

    def save_figure(self, fig: plt.Figure, filepath: str, dpi: int = 300) -> None:
        """Save figure to file"""
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        logger.info(f"Figure saved to {filepath}")

    def show_plots(self) -> None:
        """Display all plots"""
        plt.show()
