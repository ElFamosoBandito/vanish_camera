import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Camera,
  RotateCcw,
  Maximize2,
  Minimize2,
  Volume2,
  VolumeX,
  Play,
  RefreshCw,
  Download,
} from 'lucide-react';
import {
  VANISH_MODES,
  type VanishMode,
  type AppPhase,
} from './utils/vfxEngine';
import { soundEngine } from './utils/soundEffects';
import { StudioCanvas } from './components/StudioCanvas';
import { VirtualCameraGuideModal } from './components/VirtualCameraGuideModal';

export default function App() {
  const initialParams =
    typeof window !== 'undefined' ? new URLSearchParams(window.location.search) : null;
  const initialCleanMode = initialParams?.get('clean') === '1';
  const initialModeParam = (initialParams?.get('defaultEffect') as VanishMode) || 'naruto';

  const [mode, setMode] = useState<VanishMode>(
    ['normal', 'naruto', 'goku'].includes(initialModeParam)
      ? initialModeParam
      : 'naruto'
  );
  const [phase, setPhase] = useState<AppPhase>('ready');
  const [progress, setProgress] = useState<number>(0);
  const [calibrationRemainingSec, setCalibrationRemainingSec] = useState<number>(4);

  const [useRealWebcam, setUseRealWebcam] = useState<boolean>(false);
  const [webcamError, setWebcamError] = useState<string | null>(null);
  const [videoDevices, setVideoDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [mirrorVideo, setMirrorVideo] = useState<boolean>(true);
  const [soundMuted, setSoundMuted] = useState<boolean>(false);

  const [gokuHoldProgress, setGokuHoldProgress] = useState<number>(0);
  const [mediaPipeStatus, setMediaPipeStatus] = useState<string>(
    'Initialisation MediaPipe SelfieSegmenter...'
  );
  const [lastEventLog, setLastEventLog] = useState<string>(
    '3 gestes actifs simultanément : Signe V · Signe Naruto · 2 doigts front (Goku)'
  );

  const [isCleanFeedMode, setIsCleanFeedMode] = useState<boolean>(initialCleanMode);
  const [isGuideModalOpen, setIsGuideModalOpen] = useState<boolean>(false);
  const [connectedVirtualCamClients, setConnectedVirtualCamClients] = useState<number>(0);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const bgCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const animRef = useRef<number | null>(null);
  const lastActionTimeRef = useRef<number>(0);

  const activeModeDef = VANISH_MODES.find((m) => m.id === mode) || VANISH_MODES[0];

  const startVanishAnimation = useCallback((targetMode: VanishMode, reason: string) => {
    if (animRef.current) {
      cancelAnimationFrame(animRef.current);
    }

    lastActionTimeRef.current = performance.now();
    const modeConfig = VANISH_MODES.find((m) => m.id === targetMode) || VANISH_MODES[0];
    setMode(targetMode);
    setPhase('vanishing');
    setProgress(0);
    setLastEventLog(`${reason} → ${modeConfig.title}`);
    soundEngine.playModeSound(targetMode);

    const durationMs = modeConfig.durationSec * 1000;
    const start = performance.now();

    const step = (now: number) => {
      const t = Math.min(1, (now - start) / durationMs);
      setProgress(t);
      if (t < 1) {
        animRef.current = requestAnimationFrame(step);
      } else {
        lastActionTimeRef.current = performance.now();
        setPhase('vanished');
        setProgress(1);
        animRef.current = null;
      }
    };

    animRef.current = requestAnimationFrame(step);
  }, []);

  const handleResetToReady = useCallback(
    (reason: string = 'Retour dans le champ [R]') => {
      if (animRef.current) {
        cancelAnimationFrame(animRef.current);
        animRef.current = null;
      }

      lastActionTimeRef.current = performance.now();
      setPhase('reappearing');
      setProgress(0);
      setLastEventLog(reason);
      if (mode === 'naruto' || mode === 'goku') {
        soundEngine.playModeSound(mode);
      }

      const durationMs = mode === 'naruto' ? 950 : 750;
      const start = performance.now();

      const step = (now: number) => {
        const t = Math.min(1, (now - start) / durationMs);
        setProgress(t);
        if (t < 1) {
          animRef.current = requestAnimationFrame(step);
        } else {
          lastActionTimeRef.current = performance.now();
          setPhase('ready');
          setProgress(0);
          animRef.current = null;
        }
      };

      animRef.current = requestAnimationFrame(step);
    },
    [mode]
  );

  // Exact match to SPACE in vanish_cam.py:
  // Sets state.phase = "calibrating"; StudioCanvas runs the 4.0s timer and samples the median of the last 0.55s!
  const startBackgroundCalibration = useCallback(() => {
    if (animRef.current) {
      cancelAnimationFrame(animRef.current);
      animRef.current = null;
    }
    setCalibrationRemainingSec(4);
    setPhase('calibrating');
    setProgress(0);
    setLastEventLog(
      '[Webcam] Sortez du champ pendant 4 secondes : acquisition médiane du fond...'
    );
  }, []);

  const handleCalibrationComplete = useCallback(() => {
    lastActionTimeRef.current = performance.now();
    setCalibrationRemainingSec(0);
    setPhase('ready');
    setProgress(0);
    setLastEventLog('[Webcam] Fond prêt et verrouillé. Vous pouvez revenir dans le champ.');
  }, []);

  // Callback when MediaPipe detects a real gesture in the webcam stream:
  // - In `ready`: triggers vanish
  // - In `vanished`: repeating the same gesture triggers reappearance!
  // Enforces a 2.0s post-animation / 2.5s post-trigger cooldown to avoid instant re-triggering.
  const handleGestureDetected = useCallback(
    (detectedMode: VanishMode, detail: string) => {
      if (performance.now() - lastActionTimeRef.current < 1800) {
        return;
      }
      if (phase === 'ready') {
        startVanishAnimation(detectedMode, detail);
      } else if (phase === 'vanished') {
        handleResetToReady(detail);
      }
    },
    [phase, startVanishAnimation, handleResetToReady]
  );

  // Keyboard controls:
  // 1: trigger Normal Fade Out, 2: trigger Naruto, 3: trigger Goku, V: trigger vanish, R: reset ready, SPACE: calibrate background, F: fullscreen clean feed
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

      if (e.key === '1') {
        startVanishAnimation('normal', 'Touche [1] — Fade Out');
      } else if (e.key === '2') {
        startVanishAnimation('naruto', 'Touche [2] — Naruto Substitution');
      } else if (e.key === '3') {
        startVanishAnimation('goku', 'Touche [3] — Téléportation Goku');
      } else if (e.key === 'v' || e.key === 'V') {
        e.preventDefault();
        startVanishAnimation(mode, 'Touche [V] — Déclenchement manuel');
      } else if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        handleResetToReady('Touche [R] — Retour caméra');
      } else if (e.code === 'Space') {
        e.preventDefault();
        startBackgroundCalibration();
      } else if (e.key === 'f' || e.key === 'F') {
        e.preventDefault();
        setIsCleanFeedMode((v) => !v);
      } else if (e.key === 'Escape' && isCleanFeedMode) {
        setIsCleanFeedMode(false);
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [mode, isCleanFeedMode, startVanishAnimation, handleResetToReady, startBackgroundCalibration]);

  // Live WebRTC Producer Bridge for the "VanishCam Studio (Caméra Détournée)" Chrome/Brave/Edge Extension
  useEffect(() => {
    const peers = new Map<string, RTCPeerConnection>();

    const updateClientsCount = () => {
      let active = 0;
      for (const pc of peers.values()) {
        if (
          pc.connectionState === 'connected' ||
          pc.connectionState === 'connecting' ||
          pc.iceConnectionState === 'connected'
        ) {
          active++;
        }
      }
      setConnectedVirtualCamClients(active);
    };

    const sendHeartbeat = () => {
      window.postMessage(
        { source: 'VANISHCAM_STUDIO_PAGE', type: 'STUDIO_HEARTBEAT' },
        '*'
      );
    };
    sendHeartbeat();
    const hbTimer = window.setInterval(sendHeartbeat, 2500);

    const onMessage = async (event: MessageEvent) => {
      if (event.source !== window || !event.data) return;
      const msg = event.data;
      if (msg.source !== 'VANISHCAM_EXTENSION_BRIDGE') return;

      const canvas = canvasRef.current;
      if (!canvas) return;

      if (msg.type === 'REQUEST_STREAM' && msg.clientId) {
        const clientId = String(msg.clientId);
        const consumerTabId = msg.consumerTabId;

        const existing = peers.get(clientId);
        if (existing) {
          try {
            existing.close();
          } catch {}
          peers.delete(clientId);
        }

        try {
          const pc = new RTCPeerConnection({
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
          });
          peers.set(clientId, pc);

          const stream = canvas.captureStream(30);
          for (const track of stream.getVideoTracks()) {
            pc.addTrack(track, stream);
          }

          pc.onicecandidate = (iceEvent) => {
            if (iceEvent.candidate) {
              window.postMessage(
                {
                  source: 'VANISHCAM_STUDIO_PAGE',
                  type: 'WEBRTC_ICE_FROM_STUDIO',
                  clientId,
                  consumerTabId,
                  candidate: iceEvent.candidate.toJSON(),
                },
                '*'
              );
            }
          };

          pc.onconnectionstatechange = () => {
            if (
              pc.connectionState === 'closed' ||
              pc.connectionState === 'failed' ||
              pc.connectionState === 'disconnected'
            ) {
              peers.delete(clientId);
            }
            updateClientsCount();
          };

          const offer = await pc.createOffer();
          await pc.setLocalDescription(offer);

          window.postMessage(
            {
              source: 'VANISHCAM_STUDIO_PAGE',
              type: 'WEBRTC_OFFER',
              clientId,
              consumerTabId,
              sdp: pc.localDescription,
            },
            '*'
          );
          updateClientsCount();
        } catch {
          // Ignore WebRTC setup error
        }
      } else if (msg.type === 'WEBRTC_ANSWER' && msg.clientId && msg.sdp) {
        const pc = peers.get(String(msg.clientId));
        if (pc) {
          try {
            await pc.setRemoteDescription(new RTCSessionDescription(msg.sdp));
            updateClientsCount();
          } catch {}
        }
      } else if (
        msg.type === 'WEBRTC_ICE_FROM_CONSUMER' &&
        msg.clientId &&
        msg.candidate
      ) {
        const pc = peers.get(String(msg.clientId));
        if (pc) {
          try {
            await pc.addIceCandidate(new RTCIceCandidate(msg.candidate));
          } catch {}
        }
      }
    };

    window.addEventListener('message', onMessage);
    return () => {
      window.clearInterval(hbTimer);
      window.removeEventListener('message', onMessage);
      for (const pc of peers.values()) {
        try {
          pc.close();
        } catch {}
      }
      peers.clear();
    };
  }, []);

  const openCameraByDeviceId = async (deviceId?: string) => {
    try {
      setWebcamError(null);
      if (videoRef.current && videoRef.current.srcObject) {
        const prev = videoRef.current.srcObject as MediaStream;
        prev.getTracks().forEach((t) => t.stop());
      }

      const videoConstraints: MediaTrackConstraints = {
        width: { ideal: 960 },
        height: { ideal: 540 },
      };
      if (deviceId) {
        videoConstraints.deviceId = { exact: deviceId };
      }

      let stream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
        audio: false,
      });

      // Enumerate all cameras now that permission is granted
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cams = devices.filter((d) => d.kind === 'videoinput');
      setVideoDevices(cams);

      const currentTrack = stream.getVideoTracks()[0];
      const currentLabel = (currentTrack?.label || '').toLowerCase();
      const activeSettingsId = currentTrack?.getSettings()?.deviceId || deviceId || '';

      // If no specific device was requested and the browser defaulted to "OBS Virtual Camera",
      // automatically switch to the user's real physical webcam!
      if (
        !deviceId &&
        (currentLabel.includes('obs') || currentLabel.includes('virtual')) &&
        cams.length > 1
      ) {
        const physicalCam = cams.find(
          (c) =>
            !c.label.toLowerCase().includes('obs') &&
            !c.label.toLowerCase().includes('virtual')
        );
        if (physicalCam && physicalCam.deviceId) {
          stream.getTracks().forEach((t) => t.stop());
          stream = await navigator.mediaDevices.getUserMedia({
            video: {
              deviceId: { exact: physicalCam.deviceId },
              width: { ideal: 960 },
              height: { ideal: 540 },
            },
            audio: false,
          });
          setSelectedDeviceId(physicalCam.deviceId);
        } else {
          setSelectedDeviceId(activeSettingsId);
        }
      } else {
        setSelectedDeviceId(activeSettingsId);
      }

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setUseRealWebcam(true);
      startBackgroundCalibration();
    } catch {
      setWebcamError(
        'Webcam physique non détectée ou accès refusé. Le flux de démonstration reste actif.'
      );
      setUseRealWebcam(false);
    }
  };

  // Toggle Real Physical Webcam (opens at 960x540 matching vanish_cam.py and starts 4s calibration)
  const handleToggleRealWebcam = async () => {
    if (useRealWebcam) {
      if (videoRef.current && videoRef.current.srcObject) {
        const s = videoRef.current.srcObject as MediaStream;
        s.getTracks().forEach((t) => t.stop());
        videoRef.current.srcObject = null;
      }
      setUseRealWebcam(false);
      setWebcamError(null);
      setPhase('ready');
      setLastEventLog('Flux photo-réel de démonstration actif');
      return;
    }

    await openCameraByDeviceId();
  };

  const handleLaunchPiP = async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    try {
      const pipVideo = document.createElement('video');
      pipVideo.muted = true;
      pipVideo.playsInline = true;
      pipVideo.srcObject = canvas.captureStream(30);
      await pipVideo.play();
      if (document.pictureInPictureEnabled && pipVideo.requestPictureInPicture) {
        await pipVideo.requestPictureInPicture();
      }
    } catch {
      setIsCleanFeedMode(true);
    }
  };


  return (
    <div className="min-h-screen bg-[#0F172A] text-slate-100 flex flex-col">
      {/* Top Bar Contract: 1-row, 3-zone header separated by gap-8 */}
      <header className="flex items-center justify-between gap-8 px-6 py-3.5 border-b border-slate-800 bg-slate-900/80 backdrop-blur-md sticky top-0 z-30">
        <a
          href="#camera-studio"
          className="text-lg font-bold tracking-tight text-white font-display whitespace-nowrap shrink-0"
        >
          VanishCam Studio
        </a>

        <nav
          aria-label="Navigation principale"
          className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-300"
        >
          <a
            href="#camera-studio"
            className="hover:text-white hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0"
          >
            Caméra & Découpe
          </a>
          <a
            href="#modes"
            className="hover:text-white hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0"
          >
            Modes Vanish
          </a>
          <a
            href="#pipeline"
            className="hover:text-white hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0"
          >
            Fond & Masque
          </a>
          <a
            href="#obs-teams"
            onClick={(e) => {
              e.preventDefault();
              setIsGuideModalOpen(true);
            }}
            className="hover:text-white hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0"
          >
            Exporter vers Discord / Teams (OBS)
          </a>
        </nav>

        <div className="flex items-center gap-2.5 shrink-0">
          <a
            href="/vanishcam-pro-python.zip"
            download="vanishcam-pro-python.zip"
            className="flex items-center gap-1.5 px-3 py-2 text-xs font-semibold text-white bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg transition-colors whitespace-nowrap shrink-0 shadow-sm"
            title="Télécharger le script Python avec PyVirtualCam pour Discord/Teams"
          >
            <Download className="w-3.5 h-3.5 text-amber-400" />
            <span>Pack Python (.zip)</span>
          </a>
          <button
            type="button"
            onClick={() => setIsGuideModalOpen(true)}
            className="px-4 py-2 text-xs font-semibold text-slate-950 bg-amber-500 hover:bg-amber-400 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            Connecter à Discord / Teams
          </button>
        </div>
      </header>

      {/* Main Studio Workspace */}
      <main className="flex-1 max-w-[1380px] w-full mx-auto px-6 py-5 space-y-6">
        <section id="camera-studio" className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left 8 Columns: Live 960x540 Camera Feed + Direct Controls */}
          <div className="lg:col-span-8 space-y-3.5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h1 className="text-xl font-bold tracking-tight text-white font-display">
                  Vanish Webcam (Portage 1:1 de Nicovic17/vanish_webcam)
                </h1>
                <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400 mt-0.5 font-mono tabular-nums">
                  <span
                    className={
                      phase === 'vanished'
                        ? 'text-amber-400 font-semibold'
                        : 'text-emerald-400 font-semibold'
                    }
                  >
                    {phase === 'vanished'
                      ? 'Phase : vanished (Fond médian verrouillé)'
                      : phase === 'vanishing'
                      ? `Phase : vanishing (${Math.round(progress * 100)}%)`
                      : phase === 'calibrating'
                      ? `Phase : calibrating (${calibrationRemainingSec}s)`
                      : 'Phase : ready (Caméra 960×540 active)'}
                  </span>
                  <span aria-hidden="true">·</span>
                  <span>{mediaPipeStatus}</span>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                {useRealWebcam && videoDevices.length > 1 && (
                  <select
                    aria-label="Choisir la caméra"
                    value={selectedDeviceId}
                    onChange={(e) => openCameraByDeviceId(e.target.value)}
                    className="px-2.5 py-2 text-xs font-medium rounded-lg bg-slate-900 border border-slate-700 text-slate-200 focus:outline-none focus:border-amber-500 max-w-[220px] truncate cursor-pointer"
                  >
                    {videoDevices.map((dev, idx) => (
                      <option key={dev.deviceId || idx} value={dev.deviceId}>
                        {dev.label || `Caméra ${idx + 1}`}
                      </option>
                    ))}
                  </select>
                )}

                <button
                  type="button"
                  onClick={handleToggleRealWebcam}
                  className={`flex items-center gap-1.5 px-3 py-2 text-xs font-semibold rounded-lg border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                    useRealWebcam
                      ? 'bg-emerald-950/70 border-emerald-600 text-emerald-200'
                      : 'bg-amber-500 hover:bg-amber-400 border-amber-500 text-slate-950'
                  }`}
                >
                  <Camera className="w-3.5 h-3.5" />
                  <span>{useRealWebcam ? 'Webcam Réelle Active' : 'Activer ma Webcam'}</span>
                </button>

                <button
                  type="button"
                  onClick={() => setIsCleanFeedMode(true)}
                  className="flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-lg bg-slate-900 border border-slate-800 text-slate-300 hover:text-white hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                >
                  <Maximize2 className="w-3.5 h-3.5" />
                  <span>Plein Écran Propre [F]</span>
                </button>
              </div>
            </div>

            {webcamError && (
              <div
                role="status"
                className="p-3 rounded-lg bg-amber-950/50 border border-amber-800/80 text-xs text-amber-200"
              >
                {webcamError}
              </div>
            )}

            {/* Main 16:9 (960x540) Camera Output */}
            <div
              className={
                isCleanFeedMode
                  ? 'fixed inset-0 z-50 bg-black flex items-center justify-center p-0 overflow-hidden group'
                  : 'relative'
              }
            >
              <div
                className={
                  isCleanFeedMode ? 'w-full h-full max-w-[177.78vh] max-h-[56.25vw]' : 'w-full'
                }
              >
                <StudioCanvas
                  mode={mode}
                  phase={phase}
                  progress={progress}
                  calibrationRemainingSec={calibrationRemainingSec}
                  useRealWebcam={useRealWebcam}
                  mirrorVideo={mirrorVideo}
                  gokuHoldProgress={gokuHoldProgress}
                  videoRef={videoRef}
                  canvasRef={canvasRef}
                  maskCanvasRef={maskCanvasRef}
                  bgCanvasRef={bgCanvasRef}
                  onGestureDetected={handleGestureDetected}
                  onGokuHoldProgress={setGokuHoldProgress}
                  onMediaPipeStatusChange={setMediaPipeStatus}
                  onCalibrationComplete={handleCalibrationComplete}
                  onCalibrationTick={setCalibrationRemainingSec}
                />
              </div>

              {isCleanFeedMode && (
                <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-2 bg-slate-900/90 border border-slate-800 rounded-lg p-2 z-10">
                  <button
                    type="button"
                    onClick={() =>
                      phase === 'vanished'
                        ? handleResetToReady('Flux Propre')
                        : startVanishAnimation(mode, 'Flux Propre')
                    }
                    className="px-3 py-1.5 bg-amber-500 hover:bg-amber-400 text-slate-950 text-xs font-semibold rounded-md whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    {phase === 'vanished' ? 'Réapparaître [R]' : 'Disparaître [V]'}
                  </button>
                  <button
                    type="button"
                    onClick={() => setIsCleanFeedMode(false)}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium rounded-md whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    <Minimize2 className="w-3.5 h-3.5" />
                    <span>Quitter Plein Écran [Échap]</span>
                  </button>
                </div>
              )}
            </div>

            {/* Bottom Control Strip matching vanish_cam.py shortcuts: [V] Vanish, [R] Reset, [SPACE] Calibrate 4s */}
            <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800 flex flex-col xl:flex-row xl:items-center justify-between gap-3">
              <div className="min-w-0">
                <div className="text-xs font-semibold text-white whitespace-nowrap">
                  Contrôles Clavier & Gestes (identiques à vanish_cam.py)
                </div>
                <p className="text-xs text-slate-400 font-mono truncate mt-0.5">{lastEventLog}</p>
              </div>

              <div className="flex flex-wrap sm:flex-nowrap items-center gap-2 shrink-0">
                {phase === 'vanished' || phase === 'vanishing' ? (
                  <button
                    type="button"
                    data-testid="btn-trigger-reappear"
                    onClick={() => handleResetToReady('Bouton Réapparaître [R]')}
                    className="flex items-center gap-1.5 px-4 py-2 bg-emerald-500 hover:bg-emerald-400 text-slate-950 text-xs font-bold rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                    <span>Réapparaître [R]</span>
                  </button>
                ) : (
                  <button
                    type="button"
                    data-testid="btn-trigger-vanish"
                    onClick={() =>
                      startVanishAnimation(mode, `Déclenchement [${activeModeDef.title}]`)
                    }
                    className="flex items-center gap-1.5 px-4 py-2 bg-amber-500 hover:bg-amber-400 text-slate-950 text-xs font-bold rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    <Play className="w-3.5 h-3.5" />
                    <span>Lancer Vanish [V]</span>
                  </button>
                )}

                <button
                  type="button"
                  onClick={startBackgroundCalibration}
                  className="flex items-center gap-1.5 px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-medium rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span>Calibrer Fond 4s [Espace]</span>
                </button>

                <button
                  type="button"
                  onClick={() => {
                    const next = !soundMuted;
                    setSoundMuted(next);
                    soundEngine.muted = next;
                  }}
                  aria-label={soundMuted ? 'Activer le son' : 'Couper le son'}
                  className="p-2 bg-slate-950 hover:bg-slate-800 border border-slate-800 rounded-lg text-slate-300 shrink-0 cursor-pointer"
                >
                  {soundMuted ? (
                    <VolumeX className="w-4 h-4 text-red-400" />
                  ) : (
                    <Volume2 className="w-4 h-4 text-emerald-400" />
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Right 4 Columns: Coexisting Gestures + Locked Background & Mask Preview */}
          <div className="lg:col-span-4 space-y-4">
            {/* 1. Coexisting 3 Gestures (No manual selection required) */}
            <div id="modes" className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2.5">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <h2 className="text-sm font-semibold text-white font-display">
                  3 Gestes Actifs en Simultané
                </h2>
                <span className="flex items-center gap-1.5 text-xs font-mono text-emerald-400">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  Auto-détection
                </span>
              </div>

              <p className="text-xs text-slate-400 leading-relaxed">
                Aucun mode à sélectionner : faites directement l&apos;un des <strong className="text-slate-200">3 gestes</strong> devant la caméra pour déclencher son effet (et refaites un geste pour réapparaître).
              </p>

              <div className="space-y-2">
                {VANISH_MODES.map((m) => {
                  const isLastTriggered = m.id === mode;
                  return (
                    <div
                      key={m.id}
                      className={`p-2.5 rounded-lg border transition-colors ${
                        isLastTriggered
                          ? 'bg-slate-950 border-amber-500/80'
                          : 'bg-slate-950/60 border-slate-800'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-xs font-mono text-amber-400 font-semibold">
                          Geste : {m.gestureTitle}
                        </span>
                        <button
                          type="button"
                          onClick={() =>
                            startVanishAnimation(m.id, `Test manuel [${m.title}]`)
                          }
                          className="px-2 py-0.5 text-[11px] font-mono font-semibold rounded bg-slate-800 hover:bg-amber-500 hover:text-slate-950 text-slate-200 transition-colors cursor-pointer shrink-0"
                        >
                          Tester [{m.key}]
                        </button>
                      </div>
                      <div className="text-sm font-semibold text-white font-display mt-0.5">
                        {m.title}
                      </div>
                      <p className="text-xs text-slate-400 mt-0.5 leading-snug line-clamp-2">
                        {m.description}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* 2. Locked Background & Mask Preview */}
            <div
              id="pipeline"
              className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-3"
            >
              <div className="pb-2 border-b border-slate-800">
                <h2 className="text-sm font-semibold text-white font-display">
                  Acquisition du Fond (comme vanish_cam.py)
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  Appuyez sur <strong className="text-slate-200">[Espace]</strong> et sortez du champ 4s : le fond médian est capturé dans les 0.55 dernières secondes puis verrouillé.
                </p>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <div className="text-xs font-medium text-slate-300 font-mono">
                    1. Fond Médian Verrouillé
                  </div>
                  <div className="aspect-video bg-slate-950 rounded-lg overflow-hidden border border-slate-800">
                    <canvas
                      ref={bgCanvasRef}
                      width={960}
                      height={540}
                      className="w-full h-full object-cover block"
                    />
                  </div>
                </div>

                <div className="space-y-1.5">
                  <div className="text-xs font-medium text-slate-300 font-mono">
                    2. Masque (20 Hz en Vanish)
                  </div>
                  <div className="aspect-video bg-slate-950 rounded-lg overflow-hidden border border-slate-800">
                    <canvas
                      ref={maskCanvasRef}
                      width={960}
                      height={540}
                      className="w-full h-full object-cover block"
                    />
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-between pt-2 text-xs text-slate-300">
                <span className="text-slate-400 font-mono">Résolution : 960×540</span>

                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={mirrorVideo}
                    onChange={(e) => setMirrorVideo(e.target.checked)}
                    className="accent-amber-500 rounded"
                  />
                  <span>Miroir Webcam</span>
                </label>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Quiet Footer */}
      <footer className="border-t border-slate-800/80 py-3.5 px-6 text-xs text-slate-500 flex flex-wrap items-center justify-between gap-4 max-w-[1380px] w-full mx-auto">
        <span>
          Portage 1:1 de Nicovic17/vanish_webcam (vanish_cam.py & naruto_effect.py)
        </span>
        <span className="font-mono">
          3 Gestes Coexistants · [1] Fade Out · [2] Naruto · [3] Goku · [R] Réapparaître · [Espace] Calibrer 4s
        </span>
      </footer>

      <VirtualCameraGuideModal
        isOpen={isGuideModalOpen}
        onClose={() => setIsGuideModalOpen(false)}
        activeEffectId={mode}
        mirrorVideo={mirrorVideo}
        connectedVirtualCamClients={connectedVirtualCamClients}
        onEnterCleanFeedMode={() => setIsCleanFeedMode(true)}
        onLaunchPiP={handleLaunchPiP}
      />
    </div>
  );
}
