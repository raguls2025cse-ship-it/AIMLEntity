import { useCallback, useState } from 'react'
import { Upload, FileText, X, CheckCircle, AlertCircle } from 'lucide-react'

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1048576).toFixed(1)} MB`
}

export default function UploadZone({ label, subtitle, source, file, onFile, error }) {
  const [dragging, setDragging] = useState(false)

  const handleFile = useCallback((f) => {
    if (!f) return
    if (!f.name.endsWith('.tsv')) {
      onFile(null, 'Only .tsv files are accepted.')
      return
    }
    onFile(f, null)
  }, [onFile])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    handleFile(f)
  }, [handleFile])

  const onInputChange = (e) => {
    handleFile(e.target.files[0])
    e.target.value = ''
  }

  const zoneClass = [
    'upload-zone',
    dragging ? 'drag-over' : '',
    file ? 'has-file' : '',
    error ? 'has-error' : '',
  ].filter(Boolean).join(' ')

  const inputId = `file-input-${source}`

  return (
    <div className="card" style={{ height: '100%' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
        <div style={{
          width: 8, height: 8, borderRadius: '50%',
          background: source === 'source1' ? 'var(--blue-400)' :
                      source === 'source2' ? 'var(--purple-400)' : 'var(--cyan-400)'
        }} />
        <h3 style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-primary)' }}>{label}</h3>
      </div>
      <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>{subtitle}</p>

      <label htmlFor={inputId} style={{ cursor: 'pointer', display: 'block' }}>
        <div
          className={zoneClass}
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => document.getElementById(inputId).click()}
        >
          <div className="upload-icon">
            {file ? <CheckCircle size={22} color="var(--green-400)" /> :
             error ? <AlertCircle size={22} color="var(--red-400)" /> :
             <Upload size={22} color="var(--blue-400)" />}
          </div>
          <p className="upload-title">
            {file ? 'File Selected' : dragging ? 'Drop here…' : 'Click or drag a .tsv file'}
          </p>
          <p className="upload-hint">
            {file ? file.name : 'Required columns: entity_id, business_name, business_address, country'}
          </p>
        </div>
      </label>

      <input
        id={inputId}
        type="file"
        accept=".tsv"
        style={{ display: 'none' }}
        onChange={onInputChange}
      />

      {file && (
        <div className="file-info" style={{ marginTop: '0.75rem' }}>
          <FileText size={16} color="var(--green-400)" />
          <span className="file-name">{file.name}</span>
          <span className="file-size">{formatBytes(file.size)}</span>
          <button
            className="btn btn-danger btn-sm"
            style={{ padding: '0.2rem 0.5rem' }}
            onClick={(e) => { e.preventDefault(); onFile(null, null) }}
            title="Remove file"
          >
            <X size={13} />
          </button>
        </div>
      )}

      {error && (
        <div className="alert alert-error" style={{ marginTop: '0.75rem', padding: '0.6rem 0.875rem' }}>
          <AlertCircle size={14} />
          {error}
        </div>
      )}
    </div>
  )
}
