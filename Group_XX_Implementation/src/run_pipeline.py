import os
import sys
import time

# Output directory for saved visualisations (Section 8.5)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Output")

# ─── Path constants ───────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASET_DIR  = os.path.join(PROJECT_ROOT, "Dataset")
FAKE_CSV     = os.path.join(DATASET_DIR, "Fake.csv")
TRUE_CSV     = os.path.join(DATASET_DIR, "True.csv")

USE_SPARK = False   # Will be flipped to True if Spark init succeeds