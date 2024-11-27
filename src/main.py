"""Main script for signal analysis."""

import argparse
import logging
from pathlib import Path

import numpy as np

from signal_processor import ProcessingParams, SignalProcessor
from signal_visualizer import SignalVisualizer


def setup_logging(debug: bool = False) -> None:
    """Configure logging"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def read_atf_file(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Read ATF file and extract time and voltage data"""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        # Skip header (first few lines)
        for _ in range(5):
            next(f)

        for line in f:
            if line.strip():
                values = line.strip().split()
                if len(values) >= 2:
                    try:
                        time = float(values[0]) * 1000  # Convert to ms
                        voltage = float(values[1])
                        data.append((time, voltage))
                    except ValueError:
                        continue

    if not data:
        raise ValueError(f"No valid data found in {filepath}")

    time, voltage = zip(*data)
    return np.array(time), np.array(voltage)


def main():
    parser = argparse.ArgumentParser(description="Signal Analysis Tool")

    parser.add_argument("input", help="Input ATF file")
    parser.add_argument("--output", "-o", help="Output directory", default="output")
    parser.add_argument("--n", type=int, default=2, help="Number of cycles")
    parser.add_argument(
        "--t1", type=float, default=100, help="Time for hyperpolarization (ms)"
    )
    parser.add_argument(
        "--t2", type=float, default=100, help="Time for depolarization (ms)"
    )
    parser.add_argument("--V0", type=float, default=-80, help="Baseline voltage (mV)")
    parser.add_argument(
        "--V1", type=float, default=100, help="Hyperpolarization voltage (mV)"
    )
    parser.add_argument(
        "--V2", type=float, default=10, help="Depolarization voltage (mV)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Read input file
        logger.info(f"Reading file: {args.input}")
        time, voltage = read_atf_file(args.input)

        # Create processor with parameters
        params = ProcessingParams(
            n=args.n, t1=args.t1, t2=args.t2, V0=args.V0, V1=args.V1, V2=args.V2
        )

        processor = SignalProcessor(params)
        visualizer = SignalVisualizer()

        # Process signal
        logger.info("Processing signal...")
        results = processor.process_signal(voltage, time)

        # Create visualization
        logger.info("Creating visualizations...")
        fig = visualizer.create_analysis_figure(time, voltage, results)

        # Save results
        base_name = Path(args.input).stem
        fig_path = output_dir / f"{base_name}_analysis.png"
        visualizer.save_figure(fig, str(fig_path))

        # Print summary
        print("\nAnalysis Results:")
        print(f"Baseline: {results['baseline']:.2f} mV")
        print(f"Number of cycles detected: {len(results['cycles'])}")
        print(f"Integral value: {results['integral']:.6f} V·s")
        print(f"Calculated capacitance: {results['capacitance']:.2f} µF/cm²")
        print(
            f"Capacitance verification: {'PASSED' if results['capacitance_valid'] else 'FAILED'}"
        )

        # Show plots
        visualizer.show_plots()

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        if args.debug:
            raise
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
