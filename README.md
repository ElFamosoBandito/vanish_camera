# 🎥 VanishCam Studio — Caméra Virtuelle Anime & VFX par Gestes (Python & Web)

Disparaissez de votre webcam en temps réel sur **Discord**, **Microsoft Teams**, **Zoom**, **Google Meet** et **OBS Studio** d'un simple geste de la main (Goku, Naruto, Whoosh), **sans fond vert** !

---

## ⚡ Installation & Lancement en 2 Commandes (Pack Python `standalone/`)

Ouvrez votre **Terminal** dans le dossier `standalone` :

### 1️⃣ Installation (1 seule fois)

**Sur Mac / Linux :**
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

**Sur Windows :**
```bat
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt
```

### 2️⃣ Lancer la caméra

```bash
python3 vanish_cam_pro.py
```
*(Les scripts chargent automatiquement le dossier `.venv` local même si vous ouvrez un nouveau Terminal plus tard sans refaire `source .venv/bin/activate`.)*

---

## 📜 Les 4 Scripts Python Indépendants (`standalone/`)

Chaque script est **100 % indépendant** (vous pouvez lancer les 3 effets réunis ou un seul effet isolé) :

| Commande | Effet | Geste à faire devant la caméra | Touche manuelle |
| :--- | :--- | :--- | :---: |
| `python3 vanish_cam_pro.py` | ✨ **Les 3 effets réunis** | Reconnaît les **3 gestes simultanément** sans changer de mode | **`[1]`** **`[2]`** **`[3]`** |
| `python3 vanish_goku.py` | ⚡ **Téléportation Goku (*Shunkan Idō*)** | **Index + Majeur serrés posés sur le front** (hachures DBZ + découpe en bandes verticales) | **`[3]`** ou **`[G]`** |
| `python3 vanish_naruto.py` | 🪵 **Substitution Naruto (*Kawarimi no Jutsu*)** | **Mains jointes en Mudra Naruto devant le torse** (double fumée anime + bûche qui tombe avec gravité) | **`[2]`** ou **`[N]`** |
| `python3 vanish_whoosh.py` | ✌️ **Fondu Whoosh (*Fade Out / Fade In*)** | **Signe V de la victoire** (disparition progressive et fluide dans le décor) | **`[1]`** ou **`[V]`** |

> 💡 **Pour réapparaître** : Refaites simplement le geste devant la caméra lorsque vous êtes invisible, ou appuyez sur **`[R]`** !

---

## 🎮 Connecter à Discord, Teams, Zoom ou Meet (via OBS Studio)

1. Installez et ouvrez **[OBS Studio](https://obsproject.com/)**.
2. Dans OBS → **Sources** → **`+`** → **Capture d'écran macOS** (Mac) → **Capture de fenêtre** (Windows).
3. Choisissez **Fenêtre** : `[Python] VanishCam Pro`.
4. Faites un clic droit sur l'aperçu dans OBS → **Transformer** → **Adapter à l'écran** (`Cmd + F` sur Mac / `Ctrl + F` sur PC).
5. Cliquez sur **Démarrer la caméra virtuelle** en bas à droite dans OBS.
6. Dans **Discord**, **Microsoft Teams**, **Zoom** ou **Google Meet**, sélectionnez **`OBS Virtual Camera`** comme caméra !

---

## 📁 Architecture du Projet

```text
.
├── standalone/                          # 🐍 PACK PYTHON AUTONOME (Mac & Windows)
│   ├── vanish_cam_pro.py                # ✨ Les 3 effets réunis simultanément
│   ├── vanish_goku.py                   # ⚡ Téléportation Goku uniquement
│   ├── vanish_naruto.py                 # 🪵 Substitution Naruto uniquement
│   ├── vanish_whoosh.py                 # ✌️ Fondu Whoosh uniquement
│   ├── requirements.txt                 # Dépendances Python (OpenCV, MediaPipe, NumPy)
│   ├── README.md                        # Guide rapide en 2 commandes
│   └── assets/                          # Modèles MediaPipe (.task/.tflite), bûche Naruto & sons
│
├── src/                                 # 🌐 STUDIO WEB (React + TypeScript + MediaPipe WASM)
│   ├── components/
│   ├── utils/
│   └── App.tsx
│
└── public/                              # Modèles Web & pack Python téléchargeable (.zip)
```
