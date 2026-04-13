from pathlib import Path

# Chemins
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "data"  # ✅ Ajout de "src"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_eye_model.keras"

# Créer les dossiers s'ils n'existent pas
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Paramètres du modèle
IMG_SIZE = (24, 24)
LEARNING_RATE = 0.001
EPOCHS = 20
BATCH_SIZE = 32
AUGMENT = True

# Paramètres du dataset
VAL_SPLIT = 0.2
RANDOM_STATE = 42

# Paramètres de détection temps réel
EAR_THRESHOLD = 0.5
CONSEC_FRAMES = 20
DRAW_LANDMARKS = False

# Fichier son d'alarme
ALARM_SOUND = str(PROJECT_ROOT / "assets" / "alarm.wav")

# Classes
CLASS_TO_LABEL = {
    'closed': 1,
    'open': 0
}