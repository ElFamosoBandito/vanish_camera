import React, { useState } from 'react';
import { X, Copy, Check, Monitor, Layers, Download, Camera } from 'lucide-react';
import { buildCleanFeedUrl } from '../utils/vfxEngine';
import { downloadVirtualCameraExtensionZip } from '../utils/virtualCamExtensionZip';

interface VirtualCameraGuideModalProps {
  isOpen: boolean;
  onClose: () => void;
  activeEffectId: string;
  mirrorVideo: boolean;
  connectedVirtualCamClients?: number;
  onEnterCleanFeedMode: () => void;
  onLaunchPiP: () => void;
}

export const VirtualCameraGuideModal: React.FC<VirtualCameraGuideModalProps> = ({
  isOpen,
  onClose,
  activeEffectId,
  mirrorVideo,
  connectedVirtualCamClients = 0,
  onEnterCleanFeedMode,
  onLaunchPiP,
}) => {
  const [copiedUrl, setCopiedUrl] = useState(false);
  const [copiedChromeExtUrl, setCopiedChromeExtUrl] = useState(false);
  const [downloadedZip, setDownloadedZip] = useState(false);
  const [selectedTab, setSelectedTab] = useState<'python' | 'extension' | 'obs'>('python');

  if (!isOpen) return null;

  const baseUrl =
    typeof window !== 'undefined'
      ? window.location.origin + window.location.pathname
      : 'http://localhost:3000/';
  const cleanFeedUrl = buildCleanFeedUrl(baseUrl, activeEffectId, mirrorVideo);

  const handleCopyUrl = () => {
    if (typeof navigator !== 'undefined' && navigator.clipboard) {
      navigator.clipboard.writeText(cleanFeedUrl).catch(() => {});
    }
    setCopiedUrl(true);
    setTimeout(() => setCopiedUrl(false), 2500);
  };

  const handleCopyChromeExtensionsUrl = () => {
    if (typeof navigator !== 'undefined' && navigator.clipboard) {
      navigator.clipboard.writeText('chrome://extensions').catch(() => {});
    }
    setCopiedChromeExtUrl(true);
    setTimeout(() => setCopiedChromeExtUrl(false), 2500);
  };

  const handleDownloadExtension = () => {
    downloadVirtualCameraExtensionZip();
    setDownloadedZip(true);
    setTimeout(() => setDownloadedZip(false), 4000);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="virtual-cam-modal-title"
    >
      <div className="bg-slate-900 border border-slate-800 rounded-xl max-w-2xl w-full p-6 shadow-2xl text-slate-100 max-h-[90vh] overflow-y-auto">
        <div className="flex items-start justify-between gap-4 pb-4 border-b border-slate-800">
          <div>
            <h2
              id="virtual-cam-modal-title"
              className="text-xl font-semibold tracking-tight text-white font-display"
            >
              Sélectionner la « Caméra Détournée » dans Discord & Teams
            </h2>
            <p className="text-sm text-slate-400 mt-1">
              Faites apparaître <strong>🎥 VanishCam Studio (Caméra Détournée)</strong> dans la liste de choix de caméra de Discord, Microsoft Teams, Google Meet ou Zoom.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Fermer le guide caméra virtuelle"
            className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors shrink-0 cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Live WebRTC Bridge Status */}
        <div className="mt-4 flex items-center justify-between gap-3 px-3.5 py-2.5 rounded-lg bg-slate-950 border border-slate-800">
          <div className="flex items-center gap-2.5 text-xs">
            <span
              className={`w-2.5 h-2.5 rounded-full shrink-0 ${
                connectedVirtualCamClients > 0
                  ? 'bg-emerald-400 animate-pulse'
                  : 'bg-amber-400'
              }`}
            />
            <span className="font-medium text-slate-200">
              Émetteur WebRTC « Caméra Détournée » (960×540 @ 30 FPS) :{' '}
              <strong className="text-emerald-400">
                {connectedVirtualCamClients > 0
                  ? `${connectedVirtualCamClients} appel(s) Discord/Teams connecté(s) en direct`
                  : 'Prêt à diffuser vers Discord / Teams'}
              </strong>
            </span>
          </div>
        </div>

        {/* Method Selector Tabs */}
        <div className="flex items-center gap-1 p-1 bg-slate-950 border border-slate-800 rounded-lg mt-4 overflow-x-auto">
          <button
            type="button"
            onClick={() => setSelectedTab('python')}
            className={`flex-1 py-2 px-3 text-xs font-semibold rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              selectedTab === 'python'
                ? 'bg-amber-500 text-slate-950'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            ⭐ 1. Script Python Autonome (Direct Caméra Système Mac & Win)
          </button>
          <button
            type="button"
            onClick={() => setSelectedTab('extension')}
            className={`flex-1 py-2 px-3 text-xs font-semibold rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              selectedTab === 'extension'
                ? 'bg-amber-500 text-slate-950'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            2. Extension Navigateur (Teams/Discord Web)
          </button>
          <button
            type="button"
            onClick={() => setSelectedTab('obs')}
            className={`flex-1 py-2 px-3 text-xs font-semibold rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              selectedTab === 'obs'
                ? 'bg-amber-500 text-slate-950'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            3. Capture de Fenêtre OBS Studio
          </button>
        </div>

        {selectedTab === 'python' ? (
          <div className="mt-5 space-y-4">
            <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                    <Camera className="w-4 h-4 text-amber-400" />
                    <span>VanishCam Pro Autonome (Exactement comme le repo GitHub avec PyVirtualCam)</span>
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Injecte directement le flux dans la caméra système de Discord, Teams et Zoom — <strong>aucun écran ni fenêtre à capturer !</strong>
                  </p>
                </div>

                <a
                  href="/vanishcam-pro-python.zip"
                  download="vanishcam-pro-python.zip"
                  className="flex items-center gap-2 px-4 py-2.5 bg-amber-500 hover:bg-amber-400 text-slate-950 text-xs font-bold rounded-lg transition-colors whitespace-nowrap shrink-0 shadow-md"
                >
                  <Download className="w-4 h-4" />
                  <span>Télécharger le Pack Python (.zip)</span>
                </a>
              </div>

              <div className="space-y-3 pt-2 border-t border-slate-800/80">
                <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 space-y-2">
                  <div className="text-xs font-semibold text-amber-400 uppercase tracking-wider">
                    🍎 Sur Mac (2 commandes dans le Terminal — Compatible Python 3.9 à 3.14+) :
                  </div>
                  <ol className="text-xs text-slate-300 space-y-2 list-decimal list-inside">
                    <li>
                      <strong>Installation (1 seule fois) :</strong>
                      <div className="mt-1 p-2 bg-slate-950 rounded font-mono text-emerald-300 text-xs select-all">
                        python3 -m venv .venv &amp;&amp; source .venv/bin/activate &amp;&amp; pip install -r requirements.txt
                      </div>
                    </li>
                    <li>
                      <strong>Lancer la caméra :</strong>
                      <div className="mt-1 p-2 bg-slate-950 rounded font-mono text-emerald-300 text-xs select-all">
                        python3 vanish_cam_pro.py
                      </div>
                    </li>
                  </ol>
                </div>

                <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 space-y-2">
                  <div className="text-xs font-semibold text-sky-400 uppercase tracking-wider">
                    🪟 Sur Windows (2 commandes dans le Terminal / CMD) :
                  </div>
                  <div className="text-xs text-slate-300 space-y-1.5">
                    <div className="p-2 bg-slate-950 rounded font-mono text-sky-300 text-xs select-all">
                      python -m venv .venv &amp;&amp; .venv\Scripts\activate &amp;&amp; pip install -r requirements.txt
                    </div>
                    <div className="p-2 bg-slate-950 rounded font-mono text-sky-300 text-xs select-all">
                      python vanish_cam_pro.py
                    </div>
                  </div>
                </div>
              </div>

              <div className="p-3 rounded-lg bg-slate-900/90 border border-slate-800 text-xs text-slate-300 space-y-1">
                <div className="font-semibold text-white">Ce pack inclut les 4 scripts Python 100 % autonomes (avec modèles MediaPipe inclus) :</div>
                <div className="text-slate-400">
                  • <code>vanish_cam_pro.py</code> : Les 3 effets réunis simultanément (Goku + Naruto + Whoosh).<br />
                  • <code>vanish_goku.py</code> : Uniquement la Téléportation Goku (Shunkan Idō).<br />
                  • <code>vanish_naruto.py</code> : Uniquement la Substitution Naruto (bûche + fumée).<br />
                  • <code>vanish_whoosh.py</code> : Uniquement la Disparition en Fondu (Signe V).
                </div>
              </div>
            </div>
          </div>
        ) : selectedTab === 'extension' ? (
          <div className="mt-5 space-y-4">
            <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-3.5">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                    <Camera className="w-4 h-4 text-amber-400" />
                    <span>
                      Ajouter « 🎥 VanishCam Studio (Caméra Détournée) » dans le menu caméra
                    </span>
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Fonctionne directement dans <strong>Teams Web</strong>, <strong>Discord Web</strong>, <strong>Google Meet</strong> et <strong>Zoom</strong> sur Chrome, Brave et Edge — sans installer OBS.
                  </p>
                </div>

                <button
                  type="button"
                  onClick={handleDownloadExtension}
                  className="flex items-center gap-2 px-4 py-2.5 bg-amber-500 hover:bg-amber-400 text-slate-950 text-xs font-bold rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer shadow-md"
                >
                  <Download className="w-4 h-4" />
                  <span>
                    {downloadedZip
                      ? 'Extension téléchargée (.zip) !'
                      : 'Télécharger l’Extension Caméra Détournée (.zip)'}
                  </span>
                </button>
              </div>

              <ol className="space-y-2.5 text-sm text-slate-300 list-decimal list-inside pt-2 border-t border-slate-800/80">
                <li>
                  Cliquez sur le bouton jaune ci-dessus pour télécharger{' '}
                  <code className="text-amber-300 font-mono text-xs">
                    vanishcam-camera-detournee-extension.zip
                  </code>{' '}
                  puis <strong>double-cliquez dessus pour le décompresser</strong> en un dossier.
                </li>
                <li>
                  Dans votre navigateur (Chrome, Brave ou Edge), ouvrez la page des extensions :{' '}
                  <span className="inline-flex items-center gap-1.5 align-middle">
                    <code className="px-2 py-0.5 bg-slate-900 border border-slate-700 rounded text-xs font-mono text-emerald-300">
                      chrome://extensions
                    </code>
                    <button
                      type="button"
                      onClick={handleCopyChromeExtensionsUrl}
                      className="px-2 py-0.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs rounded cursor-pointer"
                    >
                      {copiedChromeExtUrl ? 'Copié !' : 'Copier'}
                    </button>
                  </span>
                </li>
                <li>
                  Activez le bouton <strong>Mode développeur</strong> (en haut à droite), puis cliquez sur <strong>Charger l’extension non empaquetée</strong> (en haut à gauche) et choisissez le dossier décompressé.
                </li>
                <li>
                  Ouvrez (ou rafraîchissez) <strong>Discord</strong> ou <strong>Microsoft Teams</strong> : au moment d’activer votre caméra, choisissez{' '}
                  <strong className="text-amber-400">
                    🎥 VanishCam Studio (Caméra Détournée)
                  </strong>{' '}
                  dans la liste des caméras !
                </li>
              </ol>

              <div className="p-3 rounded-lg bg-slate-900/90 border border-slate-800 text-xs text-slate-300">
                <strong>Comment ça marche :</strong> gardez simplement cet onglet <em>VanishCam Studio</em> ouvert avec votre webcam activée. Dès que vous faites le <strong>Signe V</strong>, le <strong>Signe Naruto</strong> ou les <strong>2 doigts sur le front (Goku)</strong>, vos interlocuteurs sur Teams/Discord voient la disparition en direct !
              </div>
            </div>
          </div>
        ) : (
          <div className="mt-5 space-y-4">
            <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-3">
              <p className="text-xs text-slate-300 leading-relaxed">
                Si vous utilisez l’<strong>application Mac ou Windows installée</strong> de Discord ou Teams (et non la version navigateur), macOS/Windows exige un pilote système pour afficher une caméra détournée dans la liste. <strong>OBS Studio</strong> installe gratuitement ce pilote officiel nommé <strong>OBS Virtual Camera</strong> :
              </p>

              <ol className="space-y-2.5 text-sm text-slate-300 list-decimal list-inside">
                <li>
                  Ici dans VanishCam Studio, activez votre webcam, calibrez votre fond (<strong>[Espace]</strong>), puis appuyez sur <strong>[F]</strong> pour passer en <em>Mode Flux Propre</em>.
                </li>
                <li>
                  Dans <strong>OBS Studio</strong> → <em>Sources</em> → <strong>+</strong> → <strong>Capture de fenêtre</strong> (ou <em>Navigateur</em> en collant l’URL ci-dessous) et sélectionnez VanishCam Studio.
                </li>
                <li>
                  Cliquez sur <strong>Démarrer la caméra virtuelle</strong> en bas à droite dans OBS.
                </li>
                <li>
                  Dans l’application Mac/PC <strong>Discord</strong> ou <strong>Microsoft Teams</strong>, cliquez sur la flèche à côté du bouton Caméra et sélectionnez la caméra détournée <strong>OBS Virtual Camera</strong>.
                </li>
              </ol>
            </div>

            <div>
              <label
                htmlFor="clean-feed-url-input"
                className="block text-xs font-medium text-slate-400 mb-1.5"
              >
                Lien direct Mode Flux Propre (16:9 sans aucun menu) :
              </label>
              <div className="flex items-center gap-2">
                <input
                  id="clean-feed-url-input"
                  type="text"
                  readOnly
                  value={cleanFeedUrl}
                  className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs font-mono text-slate-300 focus:outline-none focus:border-amber-500"
                />
                <button
                  type="button"
                  onClick={handleCopyUrl}
                  className="flex items-center gap-1.5 px-3.5 py-2 bg-slate-800 hover:bg-slate-700 text-slate-100 text-xs font-medium rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                >
                  {copiedUrl ? (
                    <Check className="w-4 h-4 text-emerald-400" />
                  ) : (
                    <Copy className="w-4 h-4" />
                  )}
                  <span>{copiedUrl ? 'Lien copié' : 'Copier l’URL'}</span>
                </button>
              </div>
            </div>
          </div>
        )}

        <div className="flex flex-wrap items-center justify-end gap-3 mt-6 pt-4 border-t border-slate-800">
          <button
            type="button"
            onClick={() => {
              onLaunchPiP();
              onClose();
            }}
            className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Layers className="w-4 h-4" />
            <span>Ouvrir en Picture-in-Picture</span>
          </button>
          <button
            type="button"
            onClick={() => {
              onEnterCleanFeedMode();
              onClose();
            }}
            className="flex items-center gap-2 px-4 py-2 bg-amber-500 hover:bg-amber-400 text-slate-950 text-xs font-semibold rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Monitor className="w-4 h-4" />
            <span>Activer le Mode Flux Propre [F]</span>
          </button>
        </div>
      </div>
    </div>
  );
};
