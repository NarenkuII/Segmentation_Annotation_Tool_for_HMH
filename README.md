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

Sur le reseau configure avec nginx, l'adresse publique est :

```text
http://hmh-segmentation.duckdns.org
```

Sur Windows, tu peux aussi lancer `run_tool.bat`.

## Workflow

1. Ouvre l'app, choisis une video, puis attends l'upload serveur.
2. Lance le scan.
3. Corrige les segments si besoin.
4. Exporte les clips. Les clips sont crees sur le PC qui lance Flask, dans `workspace/clips/`.
5. Lance `Extract COCO JSON`. Les JSON sont crees dans `workspace/keypoints/` et un lien ZIP est affiche.

Les videos uploadees, clips generes et JSON generes restent dans `workspace/`, qui est ignore par Git.

## Tracking des mains

MediaPipe tourne en mode `VIDEO`, donc il utilise le timestamp des frames au lieu de traiter chaque frame comme une photo independante. L'export JSON stabilise aussi les mains avec un `track_id` temporel. Le lissage et l'interpolation travaillent par piste, pas seulement par label `Left`/`Right`, ce qui reduit les inversions et les sauts d'une frame.

L'overlay affiche maintenant le label corrige de chaque main (`Right`/`Left`), le label brut MediaPipe quand il differe, et le cote de l'ecran (`left`/`right`). Les mains retenues par les filtres sont colorees, les autres restent grisees.

Pour normaliser le dataset sur la main droite du signeur visible a gauche de l'ecran, garde `Tracked hand -> Right only` et `Target side -> Left side`. Pour l'export JSON, garde `Hand -> Right` et `Side -> Left`.

Si la video vient d'une camera non miroir et que MediaPipe inverse gauche/droite, utilise `Hand labels -> Swap Left/Right` avant le scan ou l'export.

## CLI

```powershell
python extract_clip_keypoints.py .\mon_dossier_clips --hand Right --side left --sample-fps 10 --interpolate-gap 2 --max-jump-px 120
```

Pour inverser les labels gauche/droite :

```powershell
python extract_clip_keypoints.py .\mon_dossier_clips --hand Both --handedness-mode swap
```

## Mise en ligne

Le serveur Flask garde le calcul sur la machine qui lance `app.py`. Pour exposer l'outil a un ami, mets Flask derriere nginx avec une authentification HTTP basic et laisse Flask ecouter en local sur `127.0.0.1:8000`.

Dans la configuration actuelle, nginx tourne sur `192.168.1.16` et proxifie vers le serveur Flask sur `192.168.1.37:8000`. L'authentification HTTP basic est active sur `hmh-segmentation.duckdns.org`.

Le certificat HTTPS Let's Encrypt doit etre relance quand DuckDNS repond correctement aux requetes CAA :

```bash
sudo certbot certonly --webroot -w /var/www/html -d hmh-segmentation.duckdns.org
```
