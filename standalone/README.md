# 🎥 VanishCam Pro — Caméra Virtuelle Anime & VFX par Gestes (Python)

Disparaissez de votre webcam en temps réel sur **Discord, Microsoft Teams, Zoom, Google Meet et OBS Studio** d'un simple geste de la main, sans fond vert !

---

## ⚡ Lancement en 2 Commandes (Mac & Windows)

Ouvrez un **Terminal** dans ce dossier `standalone` :

### 1️⃣ Installation (à faire 1 seule fois)

**Sur Mac / Linux :**
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

**Sur Windows (PowerShell / CMD) :**
```bat
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt
```

### 2️⃣ Lancer la caméra

```bash
python3 vanish_cam_pro.py
```
*(Astuce : même si vous fermez le Terminal et revenez plus tard sans refaire `source .venv/bin/activate`, la commande `python3 vanish_cam_pro.py` détectera automatiquement votre dossier `.venv` !)*

---

## 📜 Les 4 Scripts Python Indépendants

Choisissez le script que vous voulez lancer selon l'effet souhaité :

| Commande | Effet inclus | Geste déclencheur | Touche manuelle |
| :--- | :--- | :--- | :---: |
| `python3 vanish_cam_pro.py` | ✨ **Les 3 effets réunis simultanément** (Goku + Naruto + Whoosh) | Reconnaît les 3 gestes en même temps | **`[1]`** / **`[2]`** / **`[3]`** |
| `python3 vanish_goku.py` | ⚡ **Téléportation Goku uniquement** (*Dragon Ball Z Shunkan Idō*) | **Index + Majeur serrés posés sur le front** | **`[3]`** ou **`[G]`** |
| `python3 vanish_naruto.py` | 🪵 **Substitution Naruto uniquement** (*Kawarimi no Jutsu* + Bûche + Fumée) | **Mains jointes en Mudra Naruto devant le torse** | **`[2]`** ou **`[N]`** |
| `python3 vanish_whoosh.py` | ✌️ **Fondu Whoosh uniquement** (*Fade Out / Fade In*) | **Signe V de la victoire** (Index + Majeur écartés) | **`[1]`** ou **`[V]`** |

> 💡 **Pour réapparaître** : Refaites simplement votre geste devant la caméra lorsque vous êtes invisible, ou appuyez sur **`[R]`** !

---

## 📸 Comment ça marche en 3 étapes ?

1. **Étape 1 — Calibrer le décor vide (`[Espace]`)** : Au lancement (ou en appuyant sur **`[Espace]`**), **sortez du champ de la caméra pendant 4 secondes**. Le programme photographie votre pièce vide.
2. **Étape 2 — Revenir devant la caméra** : Asseyez-vous normalement devant votre webcam.
3. **Étape 3 — Faire un geste !** :
   * ✌️ **Signe V** → Disparition en fondu *Whoosh*
   * 🙏 **Mains jointes (Naruto)** → Nuage de fumée + bûche qui tombe
   * ⚡ **2 doigts sur le front (Goku)** → Téléportation *Shunkan Idō* DBZ

---

## ⌨️ Raccourcis Clavier

| Touche | Action |
| :---: | :--- |
| **`[Espace]`** | **Recalibrer le décor vide (4 secondes)** — Sortez du champ pendant 4 secondes |
| **`[1]`** / **`[V]`** | Déclencher le **Fondu Whoosh** manuellement |
| **`[2]`** / **`[N]`** | Déclencher la **Substitution Naruto** manuellement |
| **`[3]`** / **`[G]`** | Déclencher la **Téléportation Goku** manuellement |
| **`[C]`** | **Changer de webcam** (pratique si OBS Virtual Camera a été sélectionnée par défaut) |
| **`[R]`** | **Réapparaître** immédiatement |
| **`[Q]`** / **`[Échap]`** | Quitter |

---

## 🎮 Utiliser dans Discord, Microsoft Teams, Zoom & Meet (via OBS Studio)

1. Installez et ouvrez **[OBS Studio](https://obsproject.com/)**.
2. Dans OBS sous **Sources**, cliquez sur **`+`** → **Capture d'écran macOS** (sur Mac) ou **Capture de fenêtre** (sur Windows) :
   * Sélectionnez **Méthode** : `Capture de fenêtre`
   * Sélectionnez la fenêtre Python **`VanishCam Pro`**
   * Faites un clic droit sur l'aperçu dans OBS → **Transformer** → **Adapter à l'écran** (`Cmd + F` sur Mac / `Ctrl + F` sur Windows).
   * Cliquez sur **Démarrer la caméra virtuelle** en bas à droite dans OBS.
3. Dans **Discord**, **Microsoft Teams**, **Zoom** ou **Google Meet**, choisissez **`OBS Virtual Camera`** comme caméra !
