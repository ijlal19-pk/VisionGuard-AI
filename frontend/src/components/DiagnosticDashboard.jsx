import { useState, useEffect } from 'react';

function CountUp({ end, decimals = 0, suffix = '' }) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    let startTimestamp = null;
    const duration = 1500; // 1.5 second animation
    let animationFrame;

    const step = (timestamp) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      
      // easeOutQuart
      const easeProgress = 1 - Math.pow(1 - progress, 4);
      
      setCount(easeProgress * end);
      
      if (progress < 1) {
        animationFrame = window.requestAnimationFrame(step);
      }
    };
    
    animationFrame = window.requestAnimationFrame(step);
    return () => window.cancelAnimationFrame(animationFrame);
  }, [end]);

  return <>{count.toFixed(decimals)}{suffix}</>;
}

function DiagnosticDashboard({ results, heatmapOpacity, previewUrl }) {
  const isGlaucoma = results.diagnosis === 'Glaucoma'
  const diagClass = isGlaucoma ? 'glaucoma' : 'normal'

  return (
    <div>
      {/* Metric Cards Row */}
      <div className="results-grid">
        <div className="metric-card">
          <div className="metric-label">Diagnosis</div>
          <div className={`metric-value ${diagClass}`}>
            {results.diagnosis}
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Confidence</div>
          <div className={`metric-value ${diagClass}`}>
            <CountUp end={results.confidence * 100} decimals={1} suffix="%" />
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Cup-to-Disc Ratio</div>
          <div className={`metric-value ${results.cup_to_disc_ratio > 0.6 ? 'warning' : 'normal'}`}>
            <CountUp end={results.cup_to_disc_ratio} decimals={2} />
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Entropy (OOD)</div>
          <div className={`metric-value ${results.entropy_score > 0.7 ? 'warning' : 'normal'}`}>
            <CountUp end={results.entropy_score} decimals={3} />
          </div>
        </div>
      </div>

      {/* Info row */}
      <div className="glass-card" style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
          <div className="control-group">
            <span className="control-label">Model Used</span>
            <span className="control-value" style={{ fontSize: '0.9rem' }}>
              {results.model_type === 'student' ? 'EfficientNet-B0 (Student)' : 'ResNet-50 (Teacher)'}
            </span>
          </div>
          <div className="control-group">
            <span className="control-label">TTA</span>
            <span className="control-value" style={{ fontSize: '0.9rem' }}>
              {results.tta_enabled ? 'Enabled (3x Augmented)' : 'Disabled'}
            </span>
          </div>
          <div className="control-group">
            <span className="control-label">Threshold</span>
            <span className="control-value" style={{ fontSize: '0.9rem' }}>
              {results.diagnostic_threshold_used}
            </span>
          </div>
          <div className="control-group">
            <span className="control-label">Glaucoma Probability</span>
            <span className="control-value" style={{ fontSize: '0.9rem' }}>
              <CountUp end={results.glaucoma_probability * 100} decimals={2} suffix="%" />
            </span>
          </div>
        </div>
      </div>

      {/* XAI Viewer: Side by Side Heatmaps */}
      <div className="glass-card" style={{ marginBottom: '1.5rem' }}>
        <div className="card-title">
          <span className="icon">&#128269;</span>
          Explainability Analysis (XAI Comparison)
        </div>

        <div className="xai-viewer">
          {/* Panel 1: Grad-CAM */}
          <div className="xai-panel">
            <div className="xai-panel-header">Grad-CAM Heatmap</div>
            <div className="xai-overlay-container">
              {previewUrl && (
                <img 
                  src={previewUrl} 
                  alt="Original" 
                  style={{ width: '100%', display: 'block' }}
                />
              )}
              {results.gradcam_heatmap && (
                <img
                  className="overlay-img"
                  src={`data:image/jpeg;base64,${results.gradcam_heatmap}`}
                  alt="Grad-CAM"
                  style={{ opacity: heatmapOpacity }}
                />
              )}
            </div>
          </div>

          {/* Panel 2: Integrated Gradients */}
          <div className="xai-panel">
            <div className="xai-panel-header">Integrated Gradients</div>
            {results.integrated_gradients_heatmap ? (
              <div className="xai-overlay-container">
                {previewUrl && (
                  <img 
                    src={previewUrl} 
                    alt="Original" 
                    style={{ width: '100%', display: 'block' }}
                  />
                )}
                <img
                  className="overlay-img"
                  src={`data:image/jpeg;base64,${results.integrated_gradients_heatmap}`}
                  alt="Integrated Gradients"
                  style={{ opacity: heatmapOpacity }}
                />
              </div>
            ) : (
              <div style={{ padding: '3rem 1rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                Captum library not available.<br />Install with: pip install captum
              </div>
            )}
          </div>


        </div>
      </div>

      {/* Recommendation Banner */}
      <div className={`recommendation ${diagClass}`}>
        <span style={{ fontSize: '1.4rem' }}>
          {isGlaucoma ? '\u26A0' : '\u2714'}
        </span>
        <span>{results.recommendation}</span>
      </div>
    </div>
  )
}

export default DiagnosticDashboard
