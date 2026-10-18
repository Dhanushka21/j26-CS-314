"""C2 Detection - central settings. Change values here, not inside other scripts."""
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------
C2_DIR = Path(__file__).resolve().parent      # .../j26-CS-314/c2_detection
REPO_ROOT = C2_DIR.parent                      # .../j26-CS-314


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contracts.models import CLASSES, FEATURES, N_FEATURES  # noqa: E402

RAW_DIR = C2_DIR / "data" / "raw" / "CSV"
TRAIN_DIR = RAW_DIR / "Train"                  # 51 CSV files
TEST_DIR = RAW_DIR / "Test"                    # 21 CSV files
PROCESSED_DIR = C2_DIR / "data" / "processed"
MODELS_DIR = C2_DIR / "models"
RESULTS_DIR = C2_DIR / "results"

# ---------------------------------------------------------------
# 2. Sliding window (LSTM)
# ---------------------------------------------------------------
WINDOW_SIZE = 10      
TRAIN_STRIDE = 5      
TEST_STRIDE = 10      

# ---------------------------------------------------------------
# 3. Sampling / balancing
# ---------------------------------------------------------------
MAX_WINDOWS_PER_CLASS_TRAIN = 10_000   
MAX_WINDOWS_PER_CLASS_TEST = 5_000
SMOTE_TARGET_PER_CLASS = 10_000        
VAL_FRACTION = 0.1                     
SEED = 42                              
# ---------------------------------------------------------------
# 4. Training (MLP + LSTM)
# ---------------------------------------------------------------
EPOCHS = int(os.getenv("C2_EPOCHS", 20))   # quick test: $env:C2_EPOCHS=2
BATCH_SIZE = 256
LR = 1e-3
PATIENCE = 4         

# ---------------------------------------------------------------
# 5. Targets
# ---------------------------------------------------------------
BASELINE_MULTICLASS_ACC = 0.733   
TARGET_WEIGHTED_F1 = 0.90         # NFR3
MAX_LATENCY_MS = 100              # NFR1

# ---------------------------------------------------------------
# 6. Feature selection
# ---------------------------------------------------------------
# IAT encodes the capture time (e.g. 84,679,164 vs 84,696,876 per file),
# not attack behaviour -> the model would learn "which capture" = leakage.
# Set env C2_KEEP_IAT=1 to run the "with IAT" ablation.
KEEP_IAT = os.getenv("C2_KEEP_IAT") == "1"
DROP_FEATURES = [] if KEEP_IAT else ["IAT"]
MODEL_FEATURES = [f for f in FEATURES if f not in DROP_FEATURES]
MODEL_FEATURE_IDX = [FEATURES.index(f) for f in MODEL_FEATURES]
RUN_TAG = "_withIAT" if KEEP_IAT else ""