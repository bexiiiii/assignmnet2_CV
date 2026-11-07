# Hand Gesture Recognition — Assignment 2 (Computer Vision, AITU)

Continuation of A1 (thumbs_up, peace, stop; 450 self-collected iPhone frames).
A1 scored 100% because test frames came from the same videos as training frames.
A2 tests the models under changed conditions and adds SOTA components:
DINOv2-reg (classifier), SAM 2.1 (hand masks), MediaPipe Gesture Recognizer and
Hand Landmarker (Google), a condition simulator and synthetic clips (Step-Video-T2V idea)
and an ArtAug-style "diagnose -> fix" refine step. Demo: own website with consent.

## Layout
```
app.py                 FastAPI server + JSON API (consent checked on every call)
web/                   website: index (consent), recognise, stress, clip, results
src/common.py          paths, classes, dataset class
src/stress.py          condition simulator, environment augmentation, SAM 2.1 masks
src/make_stress_set.py builds data/stress_test/ (SYNTHETIC) and caches masks
src/models.py          A1 CNN, DINOv2-reg classifier
src/train.py           trains logreg, cnn, cnn_env, dino, dino_env, dino_env_nobg
src/landmarks.py       MediaPipe hand landmarks + Google Gesture Recognizer
src/train_landmarks.py our MLP on hand landmarks
src/evaluate.py        accuracy / F1 per condition, figures
src/extras_light.py    latency, explanation focus, retrieval
src/eval_cascade.py    Auto = Google gesture model -> DINOv2
src/clips.py           synthetic 16-frame gesture clips + per-frame evaluation
src/explain.py         Grad-CAM, attention rollout, nearest neighbours
src/enhance.py         ArtAug-style refine
build_report.py        builds the report (docx)
results/               metrics json files, figures, screenshots
weights/               trained weights + MediaPipe task files
```

## Run (one command)
```bash
./run.sh            # sets up everything that is missing, then opens http://127.0.0.1:7860
./run.sh --force    # recompute data, training and evaluation from scratch
./run.sh --no-app   # only prepare, do not start the website
```
`run.sh` checks every step and skips it when its result already exists, so nothing is
created twice. The repository already contains the trained weights and all results,
so on a fresh clone it only installs the packages, downloads the two MediaPipe files
and starts the website. The A1 dataset (`../dataset/train|val|test/<class>/*.jpg`, or
`GESTURE_DATA=/path ./run.sh`) is needed only when something has to be rebuilt.

Needs Python 3.10-3.12 (MediaPipe has no wheels for newer versions). First start also
downloads DINOv2 (~90 MB), U2-Net (~170 MB) and SAM 2.1 hiera-tiny (~150 MB).
Tested on Apple M2, macOS. Seed 42.

## Synthetic media and privacy
Everything in data/stress_test/ except clean/ and every clip is generated from my own
frames (prefix SYN_, red SYNTHETIC label in the app). The website asks for consent
before the camera or upload works; the server keeps photos in memory only.
