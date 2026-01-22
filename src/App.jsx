import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import '@tensorflow/tfjs-backend-webgl';
import confetti from 'canvas-confetti';
import html2canvas from 'html2canvas';
import './App.css';

const CANDLE_COUNT = 6;

const App = () => {
  const [candles, setCandles] = useState(new Array(CANDLE_COUNT).fill(true));
  const [isBlown, setIsBlown] = useState(false);
  const [isCelebrating, setIsCelebrating] = useState(false);
  const [isModelReady, setIsModelReady] = useState(false);
  const [isBlowingDetected, setIsBlowingDetected] = useState(false);
  const [screenshot, setScreenshot] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState('Initializing Magic...');
  const [isAlmostPouting, setIsAlmostPouting] = useState(false);
  const [debugInfo, setDebugInfo] = useState({ ratio: 'Scanning...', sound: 0, heartbeat: 0 });
  const [isStarted, setIsStarted] = useState(false);
  const [showManualBtn, setShowManualBtn] = useState(false);

  const videoRef = useRef(null);
  const containerRef = useRef(null);
  const cameraFrameRef = useRef(null);
  const detectorRef = useRef(null);
  const frameIdRef = useRef(null);
  const analyserRef = useRef(null);
  const audioContextRef = useRef(null);

  useEffect(() => {
    const initModel = async () => {
      try {
        setLoadingMsg('Lighting the candles for the birthday star! ✨');
        await tf.setBackend('webgl');
        await tf.ready();
        const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;

        // Reverting to TFJS runtime for maximum browser compatibility
        detectorRef.current = await faceLandmarksDetection.createDetector(model, {
          runtime: 'tfjs',
          refineLandmarks: true // Enable for better lip/eye precision
        });

        setIsModelReady(true);
        setLoadingMsg('Magic Ready! Click Start.');
      } catch (err) {
        console.error('Model err:', err);
        setLoadingMsg('AI Error. Please Refresh.');
      }
    };
    initModel();
    return () => { if (frameIdRef.current) cancelAnimationFrame(frameIdRef.current); };
  }, []);

  const handleStart = async () => {
    try {
      setLoadingMsg('Waking up sensors...');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: true
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play();
          setIsStarted(true);
        };
      }

      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const analyser = audioCtx.createAnalyser();
      const mic = audioCtx.createMediaStreamSource(stream);
      analyser.fftSize = 256;
      mic.connect(analyser);
      analyserRef.current = analyser;
      audioContextRef.current = audioCtx;

      // Start the Birthday Song right away
      const audio = new Audio('/birthday_song.mp3');
      audio.loop = true;
      audio.play().catch(e => console.error("Audio autoplay blocked or failed:", e));

      // Show manual blow button after 15 seconds
      setTimeout(() => setShowManualBtn(true), 15000);
    } catch (err) {
      alert('Camera and Mic access required!');
    }
  };

  useEffect(() => {
    let active = true;
    let tick = 0;

    const detect = async () => {
      if (!isStarted || !detectorRef.current || !videoRef.current || isCelebrating || !active) {
        if (active) frameIdRef.current = requestAnimationFrame(detect);
        return;
      }

      try {
        tick = (tick + 1) % 100;

        if (audioContextRef.current && audioContextRef.current.state === 'suspended') {
          audioContextRef.current.resume();
        }

        let currentSound = 0;
        if (analyserRef.current) {
          const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
          analyserRef.current.getByteFrequencyData(dataArray);
          const sum = dataArray.reduce((prev, curr) => prev + curr, 0);
          currentSound = sum / dataArray.length;

          if (currentSound > 15 && !isBlowingDetected) {
            triggerSequence();
            return;
          }
        }

        if (videoRef.current.readyState >= 2) {
          // Use tensor for more reliable detection in tfjs runtime
          const videoTensor = tf.browser.fromPixels(videoRef.current);
          const faces = await detectorRef.current.estimateFaces(videoTensor, { flipHorizontal: false });
          videoTensor.dispose(); // Always clean up tensors!

          if (faces && faces.length > 0) {
            const face = faces[0];
            const faceWidth = face.box.width;

            // Using key points for mouth width (61, 291)
            const mL = face.keypoints[61];
            const mR = face.keypoints[291];
            if (mL && mR) {
              const mouthWidth = Math.sqrt(Math.pow(mR.x - mL.x, 2) + Math.pow(mR.y - mL.y, 2));
              const ratio = mouthWidth / faceWidth;

              setDebugInfo({ ratio: ratio.toFixed(3), sound: currentSound.toFixed(1), heartbeat: tick });

              // Pout Trigger Level
              if (ratio < 0.42 && !isBlowingDetected) {
                triggerSequence();
                return;
              }
              setIsAlmostPouting(ratio < 0.49);
            }
          } else {
            setDebugInfo(prev => ({ ...prev, ratio: 'Seeking Face...', sound: currentSound.toFixed(1), heartbeat: tick }));
            setIsAlmostPouting(false);
          }
        }
      } catch (err) {
        console.error('Detection loop err:', err);
      }

      if (active && !isBlowingDetected) {
        frameIdRef.current = requestAnimationFrame(detect);
      }
    };

    if (isStarted) detect();
    return () => { active = false; };
  }, [isStarted, isCelebrating, isBlowingDetected]);

  const triggerSequence = () => {
    setIsBlowingDetected(true);
    setIsCelebrating(true);
    setIsAlmostPouting(false);

    setIsBlown(true);
    setCandles(new Array(CANDLE_COUNT).fill(false));

    // Tiny delay to ensure state updates are considered for the capture logic if needed
    setTimeout(() => {
      captureScreenshot();
    }, 150);

    // Mega Confetti Blast
    confetti({ particleCount: 300, spread: 100, origin: { y: 0.6 }, colors: ['#FF69B4', '#00BFFF', '#FFD700', '#A364FF', '#FFFFFF'] });
    confetti({ particleCount: 150, spread: 150, origin: { y: 0.2 }, colors: ['#FF1493', '#FFD700'] });

    // Side Bursts
    setTimeout(() => {
      confetti({ particleCount: 50, angle: 60, spread: 55, origin: { x: 0 }, colors: ['#FF1493', '#FFD700'] });
      confetti({ particleCount: 50, angle: 120, spread: 55, origin: { x: 1 }, colors: ['#00BFFF', '#FFD700'] });
    }, 400);

    setTimeout(() => {
      setShowModal(true);
    }, 2500);
  };

  const captureScreenshot = async () => {
    if (!containerRef.current) return;

    try {
      const canvas = await html2canvas(containerRef.current, {
        useCORS: true,
        scale: 2,
        backgroundColor: null,
        onclone: (clonedDoc) => {
          // Hide modal in screenshot
          const modal = clonedDoc.querySelector('.modal-backdrop');
          if (modal) modal.style.display = 'none';

          // Hide UI buttons/indicators in photo
          const forceBtn = clonedDoc.querySelector('.force-blow-btn');
          if (forceBtn) forceBtn.style.display = 'none';

          const status = clonedDoc.querySelector('.overlay-status');
          if (status) status.style.display = 'none';

          const sparkles = clonedDoc.querySelectorAll('.party-sparkle');
          sparkles.forEach(s => s.style.display = 'none');
        }
      });

      setScreenshot(canvas.toDataURL('image/png'));
    } catch (err) {
      console.error("Screenshot failed:", err);
    }
  };

  const handleDownload = () => {
    const link = document.createElement('a');
    link.download = `Birthday_Moment_${Date.now()}.png`;
    link.href = screenshot;
    link.click();
  };

  const playTune = () => {
    // MP3 used instead
  };

  return (
    <div className="container" ref={containerRef}>
      {/* Party Decorations */}
      {[...Array(6)].map((_, i) => (
        <div
          key={i}
          className="party-sparkle"
          style={{
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
            fontSize: `${20 + Math.random() * 30}px`,
            animationDelay: `${Math.random() * 5}s`,
            animationDuration: `${5 + Math.random() * 5}s`
          }}
        >
          {['🎈', '✨', '🎉', '🎊', '⭐'][i % 5]}
        </div>
      ))}

      <h1 className="title" style={{ opacity: isBlown ? 0 : 1, transition: 'opacity 0.3s' }}>Happy Birthday! 🎂</h1>

      <div className="camera-frame" ref={cameraFrameRef}>
        <video ref={videoRef} className="camera-view" muted playsInline />

        {!isStarted && (
          <div className="start-overlay" onClick={handleStart}>
            <button className="btn-main" style={{ fontSize: '1.5rem', padding: '1.2rem 2.5rem' }}>
              {isModelReady ? 'START PARTY!' : loadingMsg}
            </button>
            <p className="party-hint">Grant camera/mic access to begin</p>
          </div>
        )}


        {/* Party Filter Overlay */}
        {isStarted && !isBlown && (
          <div className="party-filter"></div>
        )}

        {isStarted && (
          <div className={`overlay-status ${isBlowingDetected ? 'active' : isAlmostPouting ? 'almost' : ''}`}>
            {isBlowingDetected ? 'POOF!' : isAlmostPouting ? 'PUCKER UP!' : 'Make a wish, then blow until all the candles are out!'}
          </div>
        )}

        <div className="cake-overlay">
          <div className="cake">
            <div className="layer layer-bottom"></div>
            <div className="layer layer-middle">
              <div className="icing-drip" />
              <div className="icing-drip" />
              <div className="icing-drip" />
              <div className="icing-drip" />
              <div className="icing-drip" />
            </div>
            <div className="layer layer-top"></div>
            <div className="candles">
              {candles.map((isLit, i) => (
                <div key={i} className="candle">
                  {isLit && <div className="flame" style={{ animationDelay: `${i * 0.1}s` }}></div>}
                  {!isLit && isBlown && <div className="smoke"></div>}
                </div>
              ))}
            </div>
          </div>
        </div>

        {!isBlown && isStarted && showManualBtn && (
          <button className="force-blow-btn" onClick={triggerSequence}>
            Manual Blow ✨
          </button>
        )}
      </div>

      {showModal && (
        <div className="modal-backdrop">
          <div className="modal">
            <h3>Magical Moment Captured!</h3>
            <p className="celebration-subtitle">You are always celebrated and you deserve all the happiness in the world 💕</p>
            <img src={screenshot} className="screenshot-preview" alt="result" />
            <div className="modal-actions">
              <button className="btn-download" onClick={handleDownload}>Download Photo</button>
              <div className="btn-group">
                <button className="btn-secondary" onClick={() => window.location.reload()}>Try Again</button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;

