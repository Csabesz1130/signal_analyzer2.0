# Signal Analyzer

Signal analysis tool for processing ATF files implementing professor's algorithms.

## Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix
.\venv\Scripts\activate   # On Windows
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python src/main.py data/input/your_signal.atf
```

With parameters:
```bash
python src/main.py data/input/your_signal.atf --n 2 --t1 100 --t2 100 --V0 -80 --V1 100 --V2 10
```

## Project Structure

- src/ - Source code
- tests/ - Unit tests
- data/
  - input/ - Input ATF files
  - output/ - Analysis results
- docs/ - Documentation
