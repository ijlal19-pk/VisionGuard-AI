import { useRef, useState, useCallback } from 'react'

function ImageUploader({ onFileSelect, previewUrl, isAnalyzing }) {
  const fileInputRef = useRef(null)
  const [isDragOver, setIsDragOver] = useState(false)

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      onFileSelect(file)
    }
  }, [onFileSelect])

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false)
  }, [])

  const handleClick = useCallback(() => {
    fileInputRef.current?.click()
  }, [])

  const handleFileChange = useCallback((e) => {
    const file = e.target.files[0]
    if (file) {
      onFileSelect(file)
    }
  }, [onFileSelect])

  return (
    <div
      className={`upload-zone ${isDragOver ? 'drag-over' : ''}`}
      onClick={handleClick}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileChange}
      />

      {previewUrl ? (
        <>
          <div className="upload-preview-container">
            <img src={previewUrl} alt="Uploaded fundus" className="upload-preview" />
            {isAnalyzing && <div className="scanning-laser"></div>}
            {isAnalyzing && <div className="scanning-overlay"></div>}
          </div>
          <p style={{ marginTop: '0.5rem', color: 'var(--accent-cyan)', fontSize: '0.8rem' }}>
            {isAnalyzing ? 'Scanning retinal structures...' : 'Image loaded. Click "Analyze Image" to run diagnostics.'}
          </p>
        </>
      ) : (
        <>
          <div className="upload-icon">&#128065;</div>
          <h3>Drag & Drop Retinal Image</h3>
          <p>or click to browse your files</p>
          <p style={{ marginTop: '0.75rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Supports JPEG, PNG fundus images
          </p>
        </>
      )}
    </div>
  )
}

export default ImageUploader
