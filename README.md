# Segmentation Annotation Tool for HMH

Outil local pour segmenter une grande video de langue des signes en clips, puis exporter les keypoints de mains MediaPipe en JSON COCO-style.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Ouvre ensuite `http://127.0.0.1:8000`.

Sur Windows, tu peux aussi lancer `run_tool.bat`.

## Workflow

1. Place une video dans le dossier du projet.
2. Ouvre l'app, choisis la video, puis lance le scan.
3. Corrige les segments si besoin.
4. Exporte les clips.
5. Choisis le dossier de clips et lance `Extract COCO JSON`.

## Tracking des mains

L'export JSON stabilise maintenant les mains avec un `track_id` temporel. Le lissage et l'interpolation travaillent par piste, pas seulement par label `Left`/`Right`, ce qui reduit les inversions et les sauts d'une frame.

Si la video vient d'une camera non miroir et que MediaPipe inverse gauche/droite, utilise `Hand labels -> Swap Left/Right` avant le scan ou l'export.

## CLI

```powershell
python extract_clip_keypoints.py .\mon_dossier_clips --hand Both --sample-fps 10 --interpolate-gap 2 --max-jump-px 120
```

Pour inverser les labels gauche/droite :

```powershell
python extract_clip_keypoints.py .\mon_dossier_clips --hand Both --handedness-mode swap
```
