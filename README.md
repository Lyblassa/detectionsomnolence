
# Driver Drowsiness Detection (Python · OpenCV · MediaPipe)


Real‑time detection of early drowsiness signs using a webcam. The system computes the Eye Aspect Ratio (EAR) from MediaPipe FaceMesh landmarks and triggers a visual + audio alert when eyes remain closed for a configured number of frames.

## Utilisation rapide (démo)

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Lancer la démo webcam (pas besoin du dataset, le modèle est fourni)
python -m src.infer.realtime

## Demo
- Opens your default camera (index 0)
- Draws face/eye landmarks (optional)
- Shows live EAR
- If EAR < threshold for N consecutive frames → full‑screen red overlay + alarm sound


## Why EAR?
The Eye Aspect Ratio is a simple geometric proxy for eyelid openness. Persistent low EAR ≈ closed eyes / microsleeps.


## Install
```bash
python -m venv .venv && source .venv/bin/activate # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```


If you don’t have an alarm sound, you can generate a short beep with any WAV generator or comment out the audio line.


## Run
```bash
python -m src.main
```


## Config
Adjust thresholds in `src/config.py`:
- `EAR_THRESHOLD`: smaller → more sensitive
- `CONSEC_FRAMES`: frames under threshold to trigger alert
- `DRAW_LANDMARKS`: toggle for debugging


## Notes on Datasets (Kaggle)
This MVP is geometric‑feature based and **does not require training**. To extend with a CNN using Kaggle datasets (e.g., MRL Eye Dataset):
- Create `train/` + `val/` splits for **open/closed eyes**
- Train a small CNN (e.g., MobileNetV2 head) and replace EAR logic with classifier probabilities


## Disclaimer
This is a research/education prototype and **not a safety‑certified system**.


## License
MIT
