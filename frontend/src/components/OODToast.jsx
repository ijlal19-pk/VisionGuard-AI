function OODToast({ error, onClose }) {
  return (
    <div className="ood-toast">
      <button className="ood-toast-close" onClick={onClose}>
        &times;
      </button>
      <h4>{'\u26A0'} Out-of-Distribution Detected</h4>
      <p>{error.reason}</p>
      {error.entropy_score > 0 && (
        <p style={{ marginTop: '0.5rem', fontWeight: 600 }}>
          Entropy Score: {error.entropy_score}
        </p>
      )}
      {error.recommendation && (
        <p style={{ marginTop: '0.25rem', opacity: 0.8 }}>
          {error.recommendation}
        </p>
      )}
    </div>
  )
}

export default OODToast
