"""CLI for corrected admission-first minute-based preprocessing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maomao.data.preprocess_timeline import main, preprocess, build_admissions, assign_records

if __name__ == '__main__':
    main()
