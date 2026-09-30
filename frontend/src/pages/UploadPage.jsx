import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload, CheckCircle, AlertCircle, ArrowRight, Database } from 'lucide-react'
import UploadZone from '../components/UploadZone'
import { uploadDatasets } from '../services/api'

export default function UploadPage() {
  const navigate = useNavigate()
  const [files, setFiles] = useState({ source1: null, source2: null, source3: null })
  const [errors, setErrors] = useState({ source1: null, source2: null, source3: null })
  const [uploading, setUploading] = useState(false)
  const [uploadResult, setUploadResult] = useState(null)
  const [globalError, setGlobalError] = useState(null)

  const handleFile = (key) => (file, err) => {
    setFiles(f => ({ ...f, [key]: file }))
    setErrors(e => ({ ...e, [key]: err }))
    setUploadResult(null)
    setGlobalError(null)
  }

  const allSelected = files.source1 && files.source2 && files.source3
  const hasErrors = Object.values(errors).some(Boolean)

  const handleUpload = async () => {
    setUploading(true)
    setGlobalError(null)
    setUploadResult(null)
    try {
      const result = await uploadDatasets(files.source1, files.source2, files.source3)
      setUploadResult(result)
    } catch (err) {
      setGlobalError(err.message || 'Upload failed. Check file format.')
    } finally {
      setUploading(false)
    }
  }

  const SOURCES = [
    {
      key: 'source1',
      label: 'Source 1 — Reference Records',
      subtitle: 'Deduplicated reference business records. Entity IDs should start with S1-',
    },
    {
      key: 'source2',
      label: 'Source 2 — Business Records',
      subtitle: 'Independent source business records. Entity IDs should start with S2-',
    },
    {
      key: 'source3',
      label: 'Source 3 — Business Records',
      subtitle: 'Independent source business records. Entity IDs should start with S3-',
    },
  ]

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Upload Datasets</h1>
        <p className="page-subtitle">
          Upload all three TSV source files. Each must contain: entity_id, business_name, business_address, country.
        </p>
      </div>

      <div className="grid-3" style={{ marginBottom: '1.5rem' }}>
        {SOURCES.map(({ key, label, subtitle }) => (
          <UploadZone
            key={key}
            source={key}
            label={label}
            subtitle={subtitle}
            file={files[key]}
            error={errors[key]}
            onFile={handleFile(key)}
          />
        ))}
      </div>

      {/* Column reference */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div className="section-label">Required TSV Columns</div>
        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
          {['entity_id', 'business_name', 'business_address', 'country'].map(col => (
            <div key={col} style={{
              padding: '0.375rem 0.75rem', borderRadius: '8px',
              background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.2)',
              fontFamily: 'JetBrains Mono, monospace', fontSize: '0.8rem', color: 'var(--blue-400)'
            }}>{col}</div>
          ))}
        </div>
      </div>

      {globalError && (
        <div className="alert alert-error" style={{ marginBottom: '1rem' }}>
          <AlertCircle size={16} />
          <div><strong>Upload Error:</strong> {globalError}</div>
        </div>
      )}

      {uploadResult && (
        <div className="alert alert-success" style={{ marginBottom: '1rem', flexDirection: 'column', gap: '0.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 700 }}>
            <CheckCircle size={16} /> {uploadResult.message}
          </div>
          <div style={{ display: 'flex', gap: '1.5rem', fontSize: '0.8rem' }}>
            <span>S1: <strong>{uploadResult.statistics?.source1_records?.toLocaleString()}</strong> records</span>
            <span>S2: <strong>{uploadResult.statistics?.source2_records?.toLocaleString()}</strong> records</span>
            <span>S3: <strong>{uploadResult.statistics?.source3_records?.toLocaleString()}</strong> records</span>
          </div>
        </div>
      )}

      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
        <button
          className="btn btn-primary btn-lg"
          disabled={!allSelected || hasErrors || uploading}
          onClick={handleUpload}
          id="upload-btn"
        >
          {uploading ? (
            <><div className="spinner" style={{ width: 18, height: 18 }} /> Uploading…</>
          ) : (
            <><Upload size={18} /> Upload All Three Sources</>
          )}
        </button>

        {uploadResult && (
          <button
            className="btn btn-secondary btn-lg"
            onClick={() => navigate('/matching')}
          >
            Run Entity Matching <ArrowRight size={18} />
          </button>
        )}
      </div>
    </div>
  )
}
