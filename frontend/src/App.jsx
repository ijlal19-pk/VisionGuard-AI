import { useState, useCallback } from 'react'
import ImageUploader from './components/ImageUploader'
import DiagnosticDashboard from './components/DiagnosticDashboard'
import OODToast from './components/OODToast'

const API_BASE = 'http://127.0.0.1:8000'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [oodError, setOodError] = useState(null)

  // Settings
  const [modelType, setModelType] = useState('student')
  const [diagThreshold, setDiagThreshold] = useState(0.5)
  const [enableTTA, setEnableTTA] = useState(true)
  const [enableXAI, setEnableXAI] = useState(true)
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.6)

  const handleFileSelect = useCallback((file) => {
    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
    setResults(null)
    setOodError(null)
  }, [])

  const handleAnalyze = useCallback(async () => {
    if (!selectedFile) return

    setLoading(true)
    setOodError(null)
    setResults(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const params = new URLSearchParams({
        model_type: modelType,
        diag_threshold: diagThreshold.toString(),
        enable_tta: enableTTA.toString(),
        enable_xai_comparison: enableXAI.toString()
      })

      const response = await fetch(`${API_BASE}/api/screen?${params}`, {
        method: 'POST',
        body: formData
      })

      const data = await response.json()

      if (data.status === 'rejected') {
        setOodError(data)
        setResults(null)
      } else {
        setResults(data)
      }
    } catch (err) {
      setOodError({
        reason: 'Connection failed. Make sure the FastAPI backend is running on http://127.0.0.1:8000',
        entropy_score: 0
      })
    } finally {
      setLoading(false)
    }
  }, [selectedFile, modelType, diagThreshold, enableTTA, enableXAI])

  const handleThresholdChange = useCallback(async (newThreshold) => {
    setDiagThreshold(newThreshold)

    // If we already have results, re-analyze with the new threshold
    if (selectedFile && results) {
      setLoading(true)
      try {
        const formData = new FormData()
        formData.append('file', selectedFile)

        const params = new URLSearchParams({
          model_type: modelType,
          diag_threshold: newThreshold.toString(),
          enable_tta: enableTTA.toString(),
          enable_xai_comparison: enableXAI.toString()
        })

        const response = await fetch(`${API_BASE}/api/screen?${params}`, {
          method: 'POST',
          body: formData
        })

        const data = await response.json()
        if (data.status !== 'rejected') {
          setResults(data)
        }
      } catch (err) {
        console.error('Re-analysis failed:', err)
      } finally {
        setLoading(false)
      }
    }
  }, [selectedFile, results, modelType, enableTTA, enableXAI])

  return (
    <>
      {/* NAVBAR */}
      <nav className="navbar">
        <div className="navbar-brand">
          <div className="navbar-logo">VG</div>
          <div>
            <div className="navbar-title">VisionGuard AI</div>
            <div className="navbar-subtitle">Glaucoma Detection System</div>
          </div>
        </div>
        <div className="navbar-status">
          <span className="status-dot"></span>
          System Online
        </div>
      </nav>

      {/* MAIN CONTENT */}
      <div className="app-container">
        {/* Row 1: Upload + Controls */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
          
          {/* Left: Image Uploader */}
          <div className="glass-card">
            <div className="card-title">
              <span className="icon">&#128247;</span>
              Upload Fundus Image
            </div>
            <ImageUploader 
              onFileSelect={handleFileSelect}
              previewUrl={previewUrl}
              isAnalyzing={loading}
            />
            <div style={{ marginTop: '1rem', textAlign: 'center' }}>
              <button 
                className="btn btn-primary" 
                onClick={handleAnalyze} 
                disabled={!selectedFile || loading}
              >
                {loading ? 'Analyzing...' : 'Analyze Image'}
              </button>
            </div>
          </div>

          {/* Right: Controls Panel */}
          <div className="glass-card">
            <div className="card-title">
              <span className="icon">&#9881;</span>
              Analysis Controls
            </div>

            {/* Model Selection */}
            <div className="control-group" style={{ marginBottom: '1.25rem' }}>
              <label className="control-label">Model Selection</label>
              <select 
                className="custom-select"
                value={modelType} 
                onChange={(e) => setModelType(e.target.value)}
              >
                <option value="student">EfficientNet-B0 (Student - Fast)</option>
                <option value="teacher">ResNet-50 (Teacher - Accurate)</option>
              </select>
            </div>

            {/* Diagnostic Threshold Slider */}
            <div className="slider-container" style={{ marginBottom: '1.25rem' }}>
              <div className="slider-header">
                <span className="slider-label">Diagnostic Threshold (Strictness)</span>
                <span className="slider-value">{diagThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={diagThreshold}
                onChange={(e) => handleThresholdChange(parseFloat(e.target.value))}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                <span>High Recall (Strict)</span>
                <span>High Precision (Lenient)</span>
              </div>
            </div>

            {/* Heatmap Opacity Slider */}
            <div className="slider-container" style={{ marginBottom: '1.25rem' }}>
              <div className="slider-header">
                <span className="slider-label">Heatmap Overlay Intensity</span>
                <span className="slider-value">{Math.round(heatmapOpacity * 100)}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={heatmapOpacity}
                onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
              />
            </div>

            {/* Toggles */}
            <div style={{ display: 'flex', gap: '2rem' }}>
              <div className="toggle-container" onClick={() => setEnableTTA(!enableTTA)}>
                <div className={`toggle ${enableTTA ? 'active' : ''}`}>
                  <div className="toggle-knob"></div>
                </div>
                <span className="toggle-label">Test-Time Augmentation</span>
              </div>

              <div className="toggle-container" onClick={() => setEnableXAI(!enableXAI)}>
                <div className={`toggle ${enableXAI ? 'active' : ''}`}>
                  <div className="toggle-knob"></div>
                </div>
                <span className="toggle-label">XAI Comparison</span>
              </div>
            </div>
          </div>
        </div>

        {/* Row 2: Results */}
        {loading && (
          <div className="glass-card">
            <div className="loading-overlay">
              <div className="spinner"></div>
              <div className="loading-text">Running diagnostic pipeline with TTA and multi-XAI analysis...</div>
            </div>
          </div>
        )}

        {results && !loading && (
          <DiagnosticDashboard 
            results={results} 
            heatmapOpacity={heatmapOpacity}
            previewUrl={previewUrl}
          />
        )}
      </div>

      {/* OOD Toast */}
      {oodError && (
        <OODToast 
          error={oodError} 
          onClose={() => setOodError(null)} 
        />
      )}
    </>
  )
}

export default App
