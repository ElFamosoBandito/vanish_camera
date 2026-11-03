/**
 * Generates a complete, self-contained Chrome / Edge / Brave Extension (Manifest V3) as a downloadable .zip archive,
 * AND manages the live WebRTC Producer Bridge inside VanishCam Studio.
 *
 * When installed in Chrome/Brave/Edge, this extension registers a real selectable camera:
 *   "🎥 VanishCam Studio (Caméra Détournée)"
 * inside Discord, Microsoft Teams, Google Meet, Zoom, etc.
 * Selecting that camera streams the live 960x540 @ 30fps canvas from VanishCam Studio into the call via WebRTC.
 */

const MANIFEST_JSON = JSON.stringify(
  {
    manifest_version: 3,
    name: 'VanishCam Studio — Caméra Détournée (Discord / Teams / Meet)',
    version: '1.0.0',
    description:
      'Ajoute "VanishCam Studio (Caméra Détournée)" dans la liste des webcams de Discord, Microsoft Teams, Google Meet et Zoom.',
    permissions: ['tabs'],
    background: {
      service_worker: 'background.js',
    },
    content_scripts: [
      {
        matches: ['<all_urls>'],
        js: ['inject-virtual-cam.js'],
        run_at: 'document_start',
        world: 'MAIN',
        all_frames: true,
      },
      {
        matches: ['<all_urls>'],
        js: ['content-bridge.js'],
        run_at: 'document_start',
        world: 'ISOLATED',
        all_frames: true,
      },
    ],
  },
  null,
  2
);

const BACKGROUND_JS = `// VanishCam Studio — Background WebRTC Signaling Relay
let studioTabId = null;
let studioFrameId = null;

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (!msg) return;

  if (msg.type === 'STUDIO_HEARTBEAT') {
    studioTabId = sender.tab ? sender.tab.id : null;
    studioFrameId = typeof sender.frameId === 'number' ? sender.frameId : null;
    sendResponse({ ok: true });
    return true;
  }

  if (msg.type === 'REQUEST_STREAM') {
    const consumerTabId = sender.tab ? sender.tab.id : null;
    const consumerFrameId = typeof sender.frameId === 'number' ? sender.frameId : null;
    const payload = { ...msg, consumerTabId, consumerFrameId };
    if (studioTabId !== null) {
      const opts = studioFrameId !== null ? { frameId: studioFrameId } : {};
      chrome.tabs.sendMessage(studioTabId, payload, opts).catch(() => {
        broadcastToAllTabs(payload);
      });
    } else {
      broadcastToAllTabs(payload);
    }
    return true;
  }

  if (msg.type === 'WEBRTC_OFFER' || msg.type === 'WEBRTC_ICE_FROM_STUDIO') {
    if (msg.consumerTabId) {
      const opts = typeof msg.consumerFrameId === 'number' ? { frameId: msg.consumerFrameId } : {};
      chrome.tabs.sendMessage(msg.consumerTabId, msg, opts).catch(() => {
        broadcastToAllTabs(msg);
      });
    } else {
      broadcastToAllTabs(msg);
    }
    return true;
  }

  if (msg.type === 'WEBRTC_ANSWER' || msg.type === 'WEBRTC_ICE_FROM_CONSUMER') {
    const payload = {
      ...msg,
      consumerTabId: sender.tab ? sender.tab.id : msg.consumerTabId,
      consumerFrameId: typeof sender.frameId === 'number' ? sender.frameId : msg.consumerFrameId,
    };
    if (studioTabId !== null) {
      const opts = studioFrameId !== null ? { frameId: studioFrameId } : {};
      chrome.tabs.sendMessage(studioTabId, payload, opts).catch(() => {
        broadcastToAllTabs(payload);
      });
    } else {
      broadcastToAllTabs(payload);
    }
    return true;
  }
});

function broadcastToAllTabs(message) {
  chrome.tabs.query({}, (tabs) => {
    for (const tab of tabs) {
      if (tab.id) {
        chrome.tabs.sendMessage(tab.id, message).catch(() => {});
      }
    }
  });
}
`;

const CONTENT_BRIDGE_JS = `// VanishCam Studio — Isolated Content Script Bridge
window.addEventListener('message', (event) => {
  if (event.source !== window || !event.data) return;
  if (event.data.source === 'VANISHCAM_STUDIO_PAGE' || event.data.source === 'VANISHCAM_CONSUMER_PAGE') {
    try {
      chrome.runtime.sendMessage(event.data);
    } catch {}
  }
});

chrome.runtime.onMessage.addListener((msg) => {
  if (!msg) return;
  window.postMessage({ ...msg, source: 'VANISHCAM_EXTENSION_BRIDGE' }, '*');
});
`;

const INJECT_VIRTUAL_CAM_JS = `// VanishCam Studio — Virtual Camera Device Injection (MAIN world)
(() => {
  if (window.__VANISHCAM_INJECTED__) return;
  window.__VANISHCAM_INJECTED__ = true;

  const VIRTUAL_DEVICE_ID = 'vanishcam-virtual-camera';
  const VIRTUAL_GROUP_ID = 'vanishcam-virtual-group';
  const VIRTUAL_LABEL = '🎥 VanishCam Studio (Caméra Détournée)';

  const mediaDevices = navigator.mediaDevices;
  if (!mediaDevices) return;

  const origEnumerateDevices = mediaDevices.enumerateDevices.bind(mediaDevices);
  const origGetUserMedia = mediaDevices.getUserMedia.bind(mediaDevices);

  function createVirtualDeviceInfo() {
    const info = {
      deviceId: VIRTUAL_DEVICE_ID,
      groupId: VIRTUAL_GROUP_ID,
      kind: 'videoinput',
      label: VIRTUAL_LABEL,
      toJSON() {
        return {
          deviceId: this.deviceId,
          groupId: this.groupId,
          kind: this.kind,
          label: this.label,
        };
      },
    };
    try {
      if (typeof InputDeviceInfo !== 'undefined') {
        Object.setPrototypeOf(info, InputDeviceInfo.prototype);
      } else if (typeof MediaDeviceInfo !== 'undefined') {
        Object.setPrototypeOf(info, MediaDeviceInfo.prototype);
      }
    } catch {}
    return info;
  }

  mediaDevices.enumerateDevices = async function () {
    const devices = await origEnumerateDevices();
    // Do not inject inside the VanishCam Studio tab itself so it always picks the physical webcam
    if (document.title && document.title.includes('VanishCam Studio')) {
      return devices;
    }
    const alreadyExists = devices.some((d) => d.deviceId === VIRTUAL_DEVICE_ID);
    if (alreadyExists) return devices;
    return [createVirtualDeviceInfo(), ...devices];
  };

  function matchesVirtualDevice(videoConstraints) {
    if (!videoConstraints || typeof videoConstraints !== 'object') return false;
    const dev = videoConstraints.deviceId;
    if (!dev) return false;
    if (typeof dev === 'string') return dev === VIRTUAL_DEVICE_ID;
    if (Array.isArray(dev)) return dev.includes(VIRTUAL_DEVICE_ID);
    if (typeof dev === 'object') {
      if (dev.exact === VIRTUAL_DEVICE_ID || dev.ideal === VIRTUAL_DEVICE_ID) return true;
      if (Array.isArray(dev.exact) && dev.exact.includes(VIRTUAL_DEVICE_ID)) return true;
      if (Array.isArray(dev.ideal) && dev.ideal.includes(VIRTUAL_DEVICE_ID)) return true;
    }
    return false;
  }

  async function createVirtualCameraStream() {
    const outCanvas = document.createElement('canvas');
    outCanvas.width = 960;
    outCanvas.height = 540;
    const ctx = outCanvas.getContext('2d');

    const remoteVideo = document.createElement('video');
    remoteVideo.muted = true;
    remoteVideo.playsInline = true;
    remoteVideo.autoplay = true;

    let connected = false;
    const clientId = 'vc-' + Math.random().toString(36).slice(2, 10);

    let pc = null;

    function startWebRTC() {
      if (pc) {
        try { pc.close(); } catch {}
      }
      pc = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
      });

      pc.ontrack = (event) => {
        if (event.streams && event.streams[0]) {
          remoteVideo.srcObject = event.streams[0];
          remoteVideo.play().catch(() => {});
          connected = true;
        }
      };

      pc.onicecandidate = (event) => {
        if (event.candidate) {
          window.postMessage(
            {
              source: 'VANISHCAM_CONSUMER_PAGE',
              type: 'WEBRTC_ICE_FROM_CONSUMER',
              clientId,
              candidate: event.candidate.toJSON(),
            },
            '*'
          );
        }
      };

      pc.onconnectionstatechange = () => {
        if (pc.connectionState === 'disconnected' || pc.connectionState === 'failed') {
          connected = false;
        }
      };

      window.postMessage(
        {
          source: 'VANISHCAM_CONSUMER_PAGE',
          type: 'REQUEST_STREAM',
          clientId,
        },
        '*'
      );
    }

    window.addEventListener('message', async (event) => {
      if (event.source !== window || !event.data) return;
      const msg = event.data;
      if (msg.source !== 'VANISHCAM_EXTENSION_BRIDGE' || msg.clientId !== clientId) return;

      if (msg.type === 'WEBRTC_OFFER' && pc) {
        try {
          await pc.setRemoteDescription(new RTCSessionDescription(msg.sdp));
          const answer = await pc.createAnswer();
          await pc.setLocalDescription(answer);
          window.postMessage(
            {
              source: 'VANISHCAM_CONSUMER_PAGE',
              type: 'WEBRTC_ANSWER',
              clientId,
              sdp: pc.localDescription,
            },
            '*'
          );
        } catch {}
      } else if (msg.type === 'WEBRTC_ICE_FROM_STUDIO' && pc && msg.candidate) {
        try {
          await pc.addIceCandidate(new RTCIceCandidate(msg.candidate));
        } catch {}
      }
    });

    startWebRTC();
    const retryInterval = setInterval(() => {
      if (!connected) {
        startWebRTC();
      }
    }, 3000);

    const drawLoop = () => {
      if (connected && remoteVideo.readyState >= 2) {
        ctx.drawImage(remoteVideo, 0, 0, 960, 540);
      } else {
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, 960, 540);
        ctx.fillStyle = '#f59e0b';
        ctx.font = 'bold 26px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('🎥 VanishCam Studio — Caméra Détournée Active', 480, 240);
        ctx.fillStyle = '#94a3b8';
        ctx.font = '18px system-ui, sans-serif';
        ctx.fillText(
          'Gardez l’onglet VanishCam Studio ouvert avec la webcam activée pour recevoir le flux.',
          480,
          285
        );
      }
    };

    const renderTimer = setInterval(drawLoop, 33);
    drawLoop();

    const stream = outCanvas.captureStream(30);
    const videoTrack = stream.getVideoTracks()[0];
    if (videoTrack) {
      const origStop = videoTrack.stop.bind(videoTrack);
      videoTrack.stop = () => {
        clearInterval(renderTimer);
        clearInterval(retryInterval);
        if (pc) {
          try { pc.close(); } catch {}
        }
        origStop();
      };
    }

    return stream;
  }

  mediaDevices.getUserMedia = async function (constraints) {
    if (constraints && matchesVirtualDevice(constraints.video)) {
      const virtualVideoStream = await createVirtualCameraStream();
      if (constraints.audio) {
        try {
          const audioStream = await origGetUserMedia({ audio: constraints.audio, video: false });
          for (const audioTrack of audioStream.getAudioTracks()) {
            virtualVideoStream.addTrack(audioTrack);
          }
        } catch {}
      }
      return virtualVideoStream;
    }
    return origGetUserMedia(constraints);
  };
})();
`;

const README_TXT = `========================================================================
VANISHCAM STUDIO — EXTENSION CAMERA DETOURNEE (DISCORD / TEAMS / MEET)
========================================================================

Cette extension ajoute une vraie caméra sélectionnable nommée :
  "🎥 VanishCam Studio (Caméra Détournée)"
directement dans la liste des caméras de Discord, Microsoft Teams, Google Meet et Zoom.

INSTALLATION EN 30 SECONDES (Chrome / Brave / Edge) :
1. Décompressez ce fichier .zip dans un dossier sur votre Mac ou PC.
2. Ouvrez votre navigateur et tapez dans la barre d'adresse :
   chrome://extensions
   (ou brave://extensions ou edge://extensions)
3. Activez "Mode développeur" (Developer mode) en haut à droite.
4. Cliquez sur "Charger l'extension non empaquetée" (Load unpacked) en haut à gauche
   et sélectionnez le dossier décompressé.
5. Ouvrez ou rafraîchissez Discord, Microsoft Teams ou Google Meet :
   Quand vous activez votre caméra, choisissez "🎥 VanishCam Studio (Caméra Détournée)" !
`;

/**
 * Computes standard CRC-32 for a Uint8Array (required for valid ZIP archives).
 */
function crc32(data: Uint8Array): number {
  let crc = 0xffffffff;
  for (let i = 0; i < data.length; i++) {
    crc ^= data[i];
    for (let j = 0; j < 8; j++) {
      crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

/**
 * Packs an array of { name, content } UTF-8 text files into a standard uncompressed ZIP Blob.
 */
export function buildExtensionZipBlob(): Blob {
  const encoder = new TextEncoder();
  const files = [
    { name: 'manifest.json', data: encoder.encode(MANIFEST_JSON) },
    { name: 'background.js', data: encoder.encode(BACKGROUND_JS) },
    { name: 'content-bridge.js', data: encoder.encode(CONTENT_BRIDGE_JS) },
    { name: 'inject-virtual-cam.js', data: encoder.encode(INJECT_VIRTUAL_CAM_JS) },
    { name: 'LISEZ-MOI.txt', data: encoder.encode(README_TXT) },
  ];

  const localHeaders: Uint8Array[] = [];
  const centralHeaders: Uint8Array[] = [];
  let offset = 0;

  for (const file of files) {
    const nameBytes = encoder.encode(file.name);
    const crc = crc32(file.data);
    const size = file.data.length;

    // Local file header (30 bytes + filename)
    const localHeader = new Uint8Array(30 + nameBytes.length);
    const lv = new DataView(localHeader.buffer);
    lv.setUint32(0, 0x04034b50, true); // Local file header signature
    lv.setUint16(4, 20, true); // Version needed to extract (2.0)
    lv.setUint16(6, 0x0800, true); // UTF-8 filename flag
    lv.setUint16(8, 0, true); // Compression method (0 = store)
    lv.setUint16(10, 0, true); // Mod time
    lv.setUint16(12, 0, true); // Mod date
    lv.setUint32(14, crc, true); // CRC-32
    lv.setUint32(18, size, true); // Compressed size
    lv.setUint32(22, size, true); // Uncompressed size
    lv.setUint16(26, nameBytes.length, true); // Filename length
    lv.setUint16(28, 0, true); // Extra field length
    localHeader.set(nameBytes, 30);

    localHeaders.push(localHeader, file.data);

    // Central directory file header (46 bytes + filename)
    const centralHeader = new Uint8Array(46 + nameBytes.length);
    const cv = new DataView(centralHeader.buffer);
    cv.setUint32(0, 0x02014b50, true); // Central directory signature
    cv.setUint16(4, 20, true); // Version made by
    cv.setUint16(6, 20, true); // Version needed
    cv.setUint16(8, 0x0800, true); // UTF-8 flag
    cv.setUint16(10, 0, true); // Compression (store)
    cv.setUint16(12, 0, true); // Mod time
    cv.setUint16(14, 0, true); // Mod date
    cv.setUint32(16, crc, true); // CRC-32
    cv.setUint32(20, size, true); // Compressed size
    cv.setUint32(24, size, true); // Uncompressed size
    cv.setUint16(28, nameBytes.length, true); // Filename length
    cv.setUint16(30, 0, true); // Extra length
    cv.setUint16(32, 0, true); // Comment length
    cv.setUint16(34, 0, true); // Disk number
    cv.setUint16(36, 0, true); // Internal attrs
    cv.setUint32(38, 0, true); // External attrs
    cv.setUint32(42, offset, true); // Relative offset of local header
    centralHeader.set(nameBytes, 46);

    centralHeaders.push(centralHeader);
    offset += localHeader.length + size;
  }

  const centralDirSize = centralHeaders.reduce((acc, b) => acc + b.length, 0);
  const eocd = new Uint8Array(22);
  const ev = new DataView(eocd.buffer);
  ev.setUint32(0, 0x06054b50, true); // End of central directory signature
  ev.setUint16(4, 0, true); // Number of this disk
  ev.setUint16(6, 0, true); // Disk where central directory starts
  ev.setUint16(8, files.length, true); // Number of central directory records on this disk
  ev.setUint16(10, files.length, true); // Total number of central directory records
  ev.setUint32(12, centralDirSize, true); // Size of central directory
  ev.setUint32(16, offset, true); // Offset of start of central directory
  ev.setUint16(20, 0, true); // Comment length

  return new Blob([...localHeaders, ...centralHeaders, eocd] as unknown as BlobPart[], {
    type: 'application/zip',
  });
}

/**
 * Triggers a browser download of `vanishcam-camera-detournee-extension.zip`.
 */
export function downloadVirtualCameraExtensionZip(): void {
  const blob = buildExtensionZipBlob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'vanishcam-camera-detournee-extension.zip';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}
