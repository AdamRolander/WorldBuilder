const { useState, useEffect, useRef, useCallback } = React;

// --- Icons (unchanged) ----------------------------------------------------
const IconUpload = ({ size = 24, className = "" }) => ( <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg> );
const IconBox = ({ size = 24, className = "" }) => ( <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg> );
const IconSettings = ({ size = 24, className = "" }) => ( <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className={className}><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51z"/></svg> );
const IconDownload = ({ size = 24, className = "" }) => ( <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> );
const IconPlay = ({ size = 24, className = "" }) => ( <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className={className}><polygon points="5 3 19 12 5 21 5 3"/></svg> );
const IconServer = ({ size = 24, className = "" }) => ( <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className={className}><rect x="2" y="2" width="20" height="8" rx="2" ry="2"/><rect x="2" y="14" width="20" height="8" rx="2" ry="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg> );
const IconImage = ({ size = 24, className = "" }) => ( <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className={className}><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg> );
// VR goggles icon — only addition to the icon set.
const IconVR = ({ size = 24, className = "" }) => ( <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="M3 7h18a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1h-5l-2-3h-4l-2 3H3a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1z"/><circle cx="8" cy="12" r="1.5"/><circle cx="16" cy="12" r="1.5"/></svg> );

const WorldBuilderDashboard = () => {
  // uploadState: 'idle' | 'processing' | 'complete' | 'failed'
  const [uploadState, setUploadState] = useState('idle');
  const [activeTab, setActiveTab]     = useState('upload');
  const [progressStep, setProgressStep] = useState(0);
  const [vlm, setVlm]                 = useState({ ok: false, model: null, device: null });
  const [job, setJob]                 = useState(null);     // { job_id, scene_id }
  const [errorMsg, setErrorMsg]       = useState(null);
  const [isVRMode, setIsVRMode]       = useState(false);
  const [availableDemos, setAvailableDemos] = useState([]);
  // Demo mode: skip the real pipeline, play the animation + load a
  // pre-baked scene. Set via the picker below or via ?demo=<id>.
  const [demoMode, setDemoMode]       = useState(false);
  // Which stage-1 backend to use for the next upload ('' = server default).
  const [detector, setDetector]       = useState('');

  const fileInputRef = useRef(null);
  const viewerRef    = useRef(null);
  const supportedFormats = ['.JPG', '.JPEG', '.PNG', '.WEBP', '.BMP', '.TIFF', '.TIF'];

  // --- VLM status: poll every 5s ----------------------------------------
  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const r = await fetch('/api/vlm-status');
        const d = await r.json();
        if (!cancelled) setVlm(d);
      } catch { if (!cancelled) setVlm({ ok: false }); }
    };
    tick();
    const id = setInterval(tick, 5000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  // --- Fetch list of pre-baked demo scenes once on mount ----------------
  useEffect(() => {
    fetch('/api/demos')
      .then(r => r.json())
      .then(d => setAvailableDemos(d.demos || []))
      .catch(() => setAvailableDemos([]));
  }, []);

  // --- Job polling: when we have an active job, poll status every 800ms -
  // Skipped in demo mode — runDemo drives the UI on a fixed timer instead.
  useEffect(() => {
    if (!job || uploadState !== 'processing' || demoMode) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const r = await fetch(`/api/jobs/${job.job_id}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();
        if (cancelled) return;
        // Server returns stage 0..5; map directly to the existing progressStep.
        setProgressStep(Math.max(1, d.stage || 1));
        if (d.status === 'complete') {
          setProgressStep(5);
          setUploadState('complete');
        } else if (d.status === 'failed') {
          setErrorMsg(d.error || 'Pipeline failed');
          setUploadState('failed');
        }
      } catch (e) {
        if (!cancelled) {
          setErrorMsg(`Status check failed: ${e.message}`);
        }
      }
    };
    tick();
    const id = setInterval(tick, 800);
    return () => { cancelled = true; clearInterval(id); };
  }, [job, uploadState, demoMode]);

  // --- Upload handlers --------------------------------------------------
  const submitFile = useCallback(async (file) => {
    if (!file) return;
    setErrorMsg(null);
    setUploadState('processing');
    setProgressStep(1);
    try {
      const fd = new FormData();
      fd.append('image', file);
      if (detector) fd.append('detector', detector);
      const r = await fetch('/api/upload', { method: 'POST', body: fd });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.error || `Upload failed: HTTP ${r.status}`);
      }
      const d = await r.json();
      setJob(d);
        } catch (e) {
      setErrorMsg(e.message);
      setUploadState('failed');
    }
  }, [detector]);

  // --- Demo mode: fake the upload flow with a pre-baked scene ----------
  // Verifies the scene exists, then plays the same 5-step animation as a
  // real upload and loads the scene's viewer at the end. Used both for
  // app-only debugging and as the Thursday-demo break-glass via ?demo=.
  const runDemo = useCallback(async (sceneId) => {
    if (!sceneId) return;
    setErrorMsg(null);
    // Verify the scene actually exists before kicking off the animation —
    // catches typos in the URL param and stale picker entries.
    try {
      const r = await fetch(`/api/jobs/${sceneId}`);
      if (!r.ok) {
        setErrorMsg(`Demo scene "${sceneId}" not found in outputs/`);
        setUploadState('failed');
        return;
      }
    } catch (e) {
      setErrorMsg(`Demo lookup failed: ${e.message}`);
      setUploadState('failed');
      return;
    }
    setDemoMode(true);
    setUploadState('processing');
    setJob({ job_id: sceneId, scene_id: sceneId });
    setProgressStep(1);
    setTimeout(() => setProgressStep(2), 1200);
    setTimeout(() => setProgressStep(3), 2400);
    setTimeout(() => setProgressStep(4), 3600);
    setTimeout(() => setProgressStep(5), 4800);
    setTimeout(() => setUploadState('complete'), 6000);
  }, []);

  // ?demo=<scene_id> auto-triggers on page load. Break-glass for Thursday.
  useEffect(() => {
    const sceneId = new URLSearchParams(window.location.search).get('demo');
    if (sceneId) {
      const t = setTimeout(() => runDemo(sceneId), 400);
      return () => clearTimeout(t);
    }
  }, [runDemo]);

  const onPickFile = () => fileInputRef.current?.click();
  const onFileChange = (e) => submitFile(e.target.files?.[0]);
  const onDrop = (e) => {
    e.preventDefault();
    submitFile(e.dataTransfer.files?.[0]);
  };
  const onDragOver = (e) => e.preventDefault();

  const reset = () => {
    setUploadState('idle');
    setProgressStep(0);
    setJob(null);
    setErrorMsg(null);
    setIsVRMode(false);
    setDemoMode(false);
  };

  // --- VR Mode: fullscreen the viewer iframe ----------------------------
  // The actual WebXR session is entered by the user clicking the VR button
  // inside the viewer (added in viewer_generator.py). Fullscreen here just
  // gives the user maximum real estate to click that button cleanly.
  const toggleVR = useCallback(async () => {
    const el = viewerRef.current;
    if (!el) return;
    if (!document.fullscreenElement) {
      try {
        await el.requestFullscreen?.();
        setIsVRMode(true);
      } catch (e) {
        setErrorMsg(`Fullscreen failed: ${e.message}`);
      }
    } else {
      await document.exitFullscreen?.();
      setIsVRMode(false);
    }
  }, []);

  useEffect(() => {
    const onFsChange = () => setIsVRMode(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', onFsChange);
    return () => document.removeEventListener('fullscreenchange', onFsChange);
  }, []);

  // Append a cache-buster so iframe reloads always pull the latest viewer.html.
  // Burned cheaply on every render — fine because the iframe only mounts
  // when uploadState === 'complete'.
  const viewerUrl = React.useMemo(
    () => job?.scene_id
      ? `/outputs/${job.scene_id}/viewer.html?v=${job.scene_id}`
      : null,
    [job?.scene_id]
  );

  return (
    <div className="flex h-screen bg-brand-black text-brand-white font-sans selection:bg-brand-primary/30">

      {/* SIDEBAR NAVIGATION */}
      <div className="w-72 bg-brand-dark flex flex-col shadow-2xl relative z-20">
        <div className="p-8">
          <img
            src="/assets/logo.png"
            alt="WorldBuilder"
            className="h-10 w-auto mb-3 object-contain"
            onError={(e) => { e.currentTarget.style.display = 'none'; }}
          />
          <p className="text-xs text-brand-primary font-bold tracking-widest uppercase">photo → 3D scene · v0.2</p>
        </div>

        <nav className="flex-1 px-4 space-y-2 mt-4">
          <button
            onClick={() => setActiveTab('upload')}
            className={`w-full flex items-center space-x-3 px-4 py-3.5 rounded-xl transition-all duration-300 font-bold ${activeTab === 'upload' ? 'bg-brand-primary/20 text-brand-white border-l-4 border-brand-primary' : 'hover:bg-brand-secondary/30 text-brand-primary hover:text-brand-white border-l-4 border-transparent'}`}
          >
            <IconImage size={20} />
            <span>Generation Input</span>
          </button>

          <button
            onClick={() => setActiveTab('workspace')}
            className={`w-full flex items-center space-x-3 px-4 py-3.5 rounded-xl transition-all duration-300 font-bold ${activeTab === 'workspace' ? 'bg-brand-primary/20 text-brand-white border-l-4 border-brand-primary' : 'hover:bg-brand-secondary/30 text-brand-primary hover:text-brand-white border-l-4 border-transparent'}`}
          >
            <IconBox size={20} />
            <span>3D Workspace</span>
          </button>

          <div className="px-4 pt-4">
            <p className="text-xs text-brand-primary/70 font-bold uppercase tracking-widest mb-2">Detector</p>
            <select
              value={detector}
              onChange={(e) => setDetector(e.target.value)}
              className="w-full bg-brand-black/40 border border-brand-secondary rounded-lg px-3 py-2 text-sm text-brand-white font-bold"
            >
              <option value="">server default</option>
              <option value="qwen">local VLM (fully offline)</option>
              <option value="gemini">Gemini (cloud)</option>
            </select>
          </div>
        </nav>

        <div className="p-6 bg-brand-black/20 border-t border-brand-secondary/30">
          <a href="https://github.com/AdamRolander/WorldBuilder/blob/main/docs/INTEGRATIONS.md" target="_blank" rel="noopener"
             className="w-full flex items-center justify-center space-x-3 px-4 py-3 rounded-xl bg-brand-secondary/50 hover:bg-brand-secondary text-brand-white transition-colors border border-brand-primary/30 font-bold">
            <IconSettings size={18} />
            <span>Blender / Unreal setup</span>
          </a>
        </div>
      </div>

      {/* MAIN CONTENT AREA */}
      <div className="flex-1 flex flex-col relative overflow-hidden">
        <div className="absolute top-[-10%] right-[-5%] w-[600px] h-[600px] bg-brand-primary/10 rounded-full blur-[120px] pointer-events-none"></div>

        {/* HEADER */}
        <header className="h-24 border-b border-brand-dark flex items-center justify-between px-10 bg-brand-black/60 backdrop-blur-xl relative z-10">
          <h2 className="text-3xl font-extrabold tracking-tight text-brand-white drop-shadow-md uppercase">WORLDBUILDER</h2>
          <div className="flex items-center space-x-6">
            {/* VLM status — now actually reflects the server */}
            <div className={`flex items-center space-x-3 text-sm px-4 py-2 rounded-full border shadow-inner font-bold ${
              vlm.ok
                ? 'text-brand-primary bg-brand-dark border-brand-secondary'
                : 'text-red-300 bg-red-950/40 border-red-900'
            }`} title={vlm.model ? `${vlm.model} (${vlm.device})` : 'VLM server unreachable'}>
              <span className="relative flex h-2.5 w-2.5">
                {vlm.ok && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand-primary opacity-75"></span>}
                <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${vlm.ok ? 'bg-brand-white shadow-[0_0_8px_#FFFFFF]' : 'bg-red-500'}`}></span>
              </span>
              <span>{vlm.ok ? 'Local VLM Connected' : 'Local VLM Offline'}</span>
            </div>

            {/* VR Mode button — only meaningful when a scene is loaded */}
            <button
              onClick={toggleVR}
              disabled={uploadState !== 'complete'}
              className={`flex items-center space-x-2 px-5 py-2.5 rounded-xl font-bold tracking-wide transition-all duration-300 ${
                uploadState === 'complete'
                  ? 'bg-brand-secondary hover:bg-brand-primary text-brand-white shadow-[0_0_20px_rgba(54,125,138,0.3)] cursor-pointer'
                  : 'bg-brand-dark text-brand-primary/50 border border-brand-secondary cursor-not-allowed'
              }`}
            >
              <IconVR size={18} />
              <span>{isVRMode ? 'Exit VR' : 'VR Mode'}</span>
            </button>

            <button
              disabled={uploadState !== 'complete'}
              onClick={() => { if (job?.scene_id) window.location.href = `/api/scenes/${job.scene_id}/glb`; }}
              title="Download the whole scene as one GLB (Blender, Unreal, three.js)"
              className={`flex items-center space-x-2 px-6 py-2.5 rounded-xl font-bold tracking-wide transition-all duration-300 ${
                uploadState === 'complete'
                  ? 'bg-brand-primary hover:bg-brand-secondary text-brand-white shadow-[0_0_20px_rgba(54,125,138,0.4)] cursor-pointer'
                  : 'bg-brand-dark text-brand-primary/50 border border-brand-secondary cursor-not-allowed'
              }`}
            >
              <IconDownload size={18} />
              <span>Download GLB</span>
            </button>
          </div>
        </header>

        {/* WORKSPACE */}
        <main className="flex-1 p-10 overflow-y-auto relative z-10 flex flex-col">

          {uploadState === 'idle' && (
            <div className="flex-1 flex items-center justify-center">
              <div className="w-full max-w-4xl mx-auto">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={onFileChange}
                  className="hidden"
                />
                <div
                  className="border-2 border-dashed border-brand-secondary rounded-3xl p-16 text-center bg-brand-dark/30 hover:border-brand-primary hover:bg-brand-dark/60 transition-all duration-300 cursor-pointer group shadow-2xl relative overflow-hidden"
                  onClick={onPickFile}
                  onDrop={onDrop}
                  onDragOver={onDragOver}
                >
                  <div className="relative z-10">
                    <div className="w-24 h-24 bg-brand-dark border border-brand-secondary rounded-full flex items-center justify-center mx-auto mb-8 group-hover:scale-110 group-hover:bg-brand-secondary transition-all duration-300">
                      <IconUpload size={40} className="text-brand-white drop-shadow-[0_0_10px_rgba(255,255,255,0.3)]" />
                    </div>
                    <h3 className="text-4xl font-extrabold mb-3 text-brand-white tracking-tight">Drag & Drop Media</h3>
                    <p className="text-brand-primary mb-10 text-xl font-bold">Upload 2D source images to begin reconstruction</p>

                    <div className="flex flex-wrap justify-center gap-3 max-w-2xl mx-auto">
                      {supportedFormats.map(ext => (
                         <span key={ext} className="px-4 py-2 bg-brand-dark border border-brand-secondary rounded-xl text-sm text-brand-white font-mono font-bold shadow-inner group-hover:border-brand-primary transition-colors">
                           {ext}
                         </span>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Demo picker — stopPropagation keeps clicks here from
                    bubbling up to the parent drop zone's file-picker. */}
                {availableDemos.length > 0 && (
                  <div
                    className="mt-8 px-2"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <p className="text-xs text-brand-primary/70 font-bold uppercase tracking-widest mb-3">
                      Or load a verified demo scene
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {availableDemos.map(d => (
                        <button
                          key={d.scene_id}
                          onClick={() => runDemo(d.scene_id)}
                          className="px-4 py-2 bg-brand-dark hover:bg-brand-secondary border border-brand-secondary hover:border-brand-primary rounded-lg text-xs text-brand-white font-mono font-bold transition-colors"
                        >
                          {d.scene_id}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {uploadState === 'processing' && (
            <div className="flex-1 flex flex-col items-center justify-center">
              <div className="w-20 h-20 relative mx-auto mb-10">
                <div className="absolute inset-0 border-4 border-brand-dark rounded-full"></div>
                <div className="absolute inset-0 border-4 border-brand-primary rounded-full border-t-transparent border-l-transparent animate-spin"></div>
                <div className="absolute inset-2 border-4 border-brand-secondary rounded-full border-b-transparent border-r-transparent animate-[spin_1.5s_linear_infinite_reverse]"></div>
              </div>

              <h3 className="text-3xl font-extrabold mb-10 text-brand-white">Pipeline Generation in Progress</h3>

              <div className="w-full max-w-lg mx-auto bg-brand-dark/80 p-8 rounded-2xl border border-brand-secondary shadow-2xl relative overflow-hidden">
                <div className="absolute top-0 left-0 w-2 bg-brand-primary h-full"></div>
                <div className="space-y-6 pl-6 font-bold">
                  {['Object detection (VLM)...', 'SAM 3 segmentation...', 'SAM 3D reconstruction...', 'Room layout & placement...', 'Viewer & exports...'].map((step, idx) => (
                    <div key={step} className={`flex items-center space-x-4 transition-opacity duration-500 ${progressStep > idx ? 'opacity-100' : 'opacity-40'}`}>
                      {progressStep > idx + 1 ? <IconPlay size={20} className="fill-brand-primary text-brand-primary" /> : (progressStep === idx + 1 ? <div className="w-5 h-5 rounded-full border-4 border-brand-primary animate-pulse"></div> : <div className="w-5 h-5 rounded-full border-4 border-brand-secondary"></div>)}
                      <span className="text-lg text-brand-white">{step}</span>
                    </div>
                  ))}
                </div>
              </div>

              {job?.scene_id && (
                <p className="mt-6 text-xs text-brand-primary/70 font-mono">
                  scene: {job.scene_id}
                  {demoMode && <span className="ml-3 px-2 py-0.5 bg-brand-primary/20 text-brand-primary rounded">DEMO</span>}
                </p>
              )}
            </div>
          )}

          {uploadState === 'failed' && (
            <div className="flex-1 flex flex-col items-center justify-center">
              <div className="max-w-2xl text-center bg-brand-dark/80 p-10 rounded-3xl border border-red-900 shadow-2xl">
                <h3 className="text-3xl font-extrabold text-red-300 mb-4">Pipeline Failed</h3>
                <p className="text-brand-white/80 font-mono text-sm mb-8 break-words">{errorMsg || 'Unknown error'}</p>
                <button
                  onClick={reset}
                  className="px-6 py-2.5 rounded-xl bg-brand-primary hover:bg-brand-secondary text-brand-white font-bold"
                >
                  Try Again
                </button>
              </div>
            </div>
          )}

          {uploadState === 'complete' && viewerUrl && (
            <div className="h-full flex flex-col animate-[fadeIn_0.5s_ease-in-out]">
              <div
                ref={viewerRef}
                className="flex-1 bg-[#010001] rounded-3xl border border-brand-secondary relative overflow-hidden shadow-[inset_0_0_50px_rgba(19,51,54,0.3)]"
              >
                <iframe
                  src={viewerUrl}
                  title="3D Scene Viewer"
                  className="w-full h-full border-0 rounded-3xl"
                  allow="xr-spatial-tracking; fullscreen; gyroscope; accelerometer"
                  allowFullScreen
                />
                <div className="absolute left-6 bottom-6 z-10 bg-brand-dark/90 border border-brand-secondary rounded-xl px-4 py-2 text-xs font-mono text-brand-primary">
                  {job?.scene_id}
                </div>
                <button
                  onClick={reset}
                  className="absolute right-6 bottom-6 z-10 bg-brand-dark/90 hover:bg-brand-secondary border border-brand-secondary rounded-xl px-4 py-2 text-xs font-bold text-brand-white"
                >
                  New Scene
                </button>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<WorldBuilderDashboard />);