#!/usr/bin/env bash
# One command for everything:   ./run.sh            (skips every step that is already done)
#                                ./run.sh --force    (recomputes data, training and evaluation)
#                                ./run.sh --no-app   (only prepare, do not start the website)
# Every step checks whether its output already exists, so nothing is created twice.
set -euo pipefail
cd "$(dirname "$0")"

FORCE=0; APP=1
for a in "$@"; do
  case "$a" in
    --force) FORCE=1 ;;
    --no-app) APP=0 ;;
    *) echo "unknown option: $a"; exit 1 ;;
  esac
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$1"; }
skip() { printf '   skip: %s\n' "$1"; }
have() { [ "$FORCE" = 0 ] && [ -e "$1" ]; }
PY=.venv/bin/python
export GESTURE_DEVICE="${GESTURE_DEVICE:-cpu}"   # MPS hung on an 8 GB Mac, CPU is safe everywhere

# ---------------------------------------------------------------- 1. Python environment
say "1/8 Python environment"
if [ ! -x "$PY" ]; then
  BASE=""
  for p in python3.12 python3.11 python3.10; do command -v "$p" >/dev/null && { BASE=$p; break; }; done
  [ -z "$BASE" ] && { echo "Python 3.10-3.12 is needed (MediaPipe has no wheels for newer versions)."; exit 1; }
  "$BASE" -m venv .venv
fi
STAMP=.venv/.requirements.sha
NEW=$( (shasum requirements.txt 2>/dev/null || sha1sum requirements.txt) | cut -d' ' -f1)
if [ "$(cat "$STAMP" 2>/dev/null)" != "$NEW" ]; then
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r requirements.txt
  echo "$NEW" > "$STAMP"
else
  skip "packages already installed"
fi

# ---------------------------------------------------------------- 2. MediaPipe model files
say "2/8 MediaPipe model files"
MP=https://storage.googleapis.com/mediapipe-models
for f in hand_landmarker gesture_recognizer; do
  if [ -s "weights/$f.task" ]; then skip "weights/$f.task"
  else curl -fsSL -o "weights/$f.task" "$MP/$f/$f/float16/latest/$f.task"; echo "   downloaded $f.task"; fi
done

# ---------------------------------------------------------------- dataset (only needed to rebuild)
DATA="${GESTURE_DATA:-../dataset}"
HAS_DATA=0; [ -d "$DATA/train" ] && [ -d "$DATA/test" ] && HAS_DATA=1
need_rebuild=0
for f in results/metrics.json results/landmark_metrics.json results/cascade_metrics.json results/clips_metrics.json \
         weights/cnn.pt weights/dino_env.pt weights/landmark_mlp.pt results/bank_dino_env.pt results/landmarks_cache.npz; do
  [ -e "$f" ] || need_rebuild=1
done
[ "$FORCE" = 1 ] && need_rebuild=1

if [ "$need_rebuild" = 1 ] && [ "$HAS_DATA" = 0 ]; then
  echo "The A1 dataset was not found at $DATA (train/val/test/<class>/*.jpg)."
  echo "Put it there or run:  GESTURE_DATA=/path/to/dataset ./run.sh"
  exit 1
fi

if [ "$need_rebuild" = 0 ]; then
  say "3-7/8 Data, training, evaluation"
  skip "all results and weights are already here (use --force to recompute)"
else
  export GESTURE_DATA="$DATA"
  # ------------------------------------------------------------ 3. stress test set + hand masks
  say "3/8 Synthetic stress test set (8 conditions x 69 images)"
  n=$(find data/stress_test -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')
  if [ "$FORCE" = 0 ] && [ "$n" -ge 552 ]; then skip "data/stress_test ($n images)"
  else "$PY" -u src/make_stress_set.py; fi

  # ------------------------------------------------------------ 4. training (only missing models)
  say "4/8 Training"
  jobs=""
  for m in logreg cnn cnn_env dino dino_env dino_env_nobg; do
    w=weights/$m.pt; [ "$m" = logreg ] && w=weights/logreg.pkl
    if have "$w"; then skip "$w"; else jobs="$jobs $m"; fi
  done
  [ -n "$jobs" ] && "$PY" -u src/train.py $jobs

  say "5/8 Hand-landmark classifier"
  if have weights/landmark_mlp.pt && have results/landmark_metrics.json; then skip "weights/landmark_mlp.pt"
  else rm -f results/landmarks_cache.npz; "$PY" -u src/train_landmarks.py; fi

  # ------------------------------------------------------------ 6. evaluation
  say "6/8 Evaluation"
  if have results/per_condition.json && have results/preds.npz; then skip "per-condition accuracy"
  else "$PY" -u src/evaluate.py; fi
  if have results/metrics.json && grep -q hand_focus results/metrics.json; then skip "latency / explanation / retrieval"
  else rm -f results/metrics.json; "$PY" -u src/extras_light.py; fi
  if have results/cascade_metrics.json; then skip "Auto cascade"; else "$PY" -u src/eval_cascade.py; fi
  if have results/explain_grid.png; then skip "figures"; else "$PY" -u src/figures_extra.py; fi

  # ------------------------------------------------------------ 7. synthetic clips
  say "7/8 Synthetic clips"
  if have results/clips_metrics.json; then skip "results/clips_metrics.json"; else "$PY" -u src/clips.py; fi
fi

# ---------------------------------------------------------------- 8. website
if [ "$APP" = 1 ]; then
  say "8/8 Website"
  if lsof -i :7860 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "   already running: http://127.0.0.1:7860"
  else
    echo "   starting http://127.0.0.1:7860  (Ctrl+C to stop)"
    exec "$PY" app.py
  fi
fi
