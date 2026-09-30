import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { GitBranch, Play, BarChart3, AlertCircle, CheckCircle2 } from 'lucide-react'
import ProgressPanel from '../components/ProgressPanel'
import { runEntityMatching, getMatchingStatus } from '../services/api'

export default function MatchingPage() {
  const navigate = useNavigate()
  const [jobId, setJobId] = useState(null)
  const [jobStatus, setJobStatus] = useState(null)
  const [error, setError] = useState(null)
  const [running, setRunning] = useState(false)
  const pollRef = useRef(null)

  const startPolling = (id) => {
    if (pollRef.current) clearInterval(pollRef.current)
    pollRef.current = setInterval(async () => {
      try {
        const status = await getMatchingStatus(id)
        setJobStatus(status)
        if (status.status === 'completed' || status.status === 'failed') {
          clearInterval(pollRef.current)
          setRunning(false)
        }
      } catch (e) {
        setError('Failed to poll job status: ' + e.message)
        clearInterval(pollRef.current)
        setRunning(false)
      }
    }, 2000)
  }

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current) }, [])

  const handleStart = async () => {
    setError(null)
    setJobStatus(null)
    setRunning(true)
    try {
      const res = await runEntityMatching()
      setJobId(res.job_id)
      startPolling(res.job_id)
    } catch (e) {
      setError(e.message || 'Failed to start matching.')
      setRunning(false)
    }
  }

  const completed = jobStatus?.status === 'completed'
  const failed = jobStatus?.status === 'failed'

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Entity Matching</h1>
        <p className="page-subtitle">
          Run the full ML entity resolution pipeline on the uploaded datasets.
        </p>
      </div>

      {/* Info card */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div className="section-label">Pipeline Overview</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
          {[
            { label: 'Model', value: 'HistGradientBoosting', color: 'var(--blue-400)' },
            { label: 'Threshold', value: '0.70', color: 'var(--purple-400)' },
            { label: 'Precision', value: '99.53%', color: 'var(--green-400)' },
            { label: 'Recall', value: '97.64%', color: 'var(--cyan-400)' },
            { label: 'Macro F0.5', value: '98.98%', color: 'var(--yellow-400)' },
            { label: 'Blocking Recall', value: '99.75%', color: 'var(--purple-400)' },
          ].map(({ label, value, color }) => (
            <div key={label} style={{
              padding: '0.875rem 1rem', borderRadius: '10px',
              background: 'rgba(10, 22, 40, 0.5)', border: '1px solid var(--border)'
            }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '0.25rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{label}</div>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, color }}>{value}</div>
            </div>
          ))}
        </div>
      </div>

      {error && (
        <div className="alert alert-error" style={{ marginBottom: '1rem' }}>
          <AlertCircle size={16} />
          <div>
            <strong>Error:</strong> {error}
            {error.includes('upload') || error.includes('files') ? (
              <div style={{ marginTop: '0.375rem', fontSize: '0.8rem' }}>
                Please upload all three source files on the{' '}
                <a href="/upload" style={{ color: 'inherit', textDecoration: 'underline' }}>Upload page</a> first.
              </div>
            ) : null}
          </div>
        </div>
      )}

      {/* Start button */}
      <div style={{ marginBottom: '1.5rem' }}>
        <button
          className="btn btn-primary btn-lg"
          disabled={running}
          onClick={handleStart}
          id="start-matching-btn"
        >
          {running ? (
            <><div className="spinner" style={{ width: 18, height: 18 }} /> Running…</>
          ) : (
            <><Play size={18} fill="white" /> Start Entity Resolution</>
          )}
        </button>
        {jobId && (
          <span style={{ marginLeft: '1rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Job ID: <span className="mono" style={{ color: 'var(--blue-400)' }}>{jobId}</span>
          </span>
        )}
      </div>

      {/* Progress */}
      {jobStatus && (
        <div className="card" style={{ marginBottom: '1.5rem' }}>
          <ProgressPanel jobStatus={jobStatus} />
        </div>
      )}

      {/* Navigation after completion */}
      {completed && (
        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
          <button className="btn btn-primary" onClick={() => navigate('/results')}>
            <BarChart3 size={16} /> View Results
          </button>
          <button className="btn btn-ghost" onClick={handleStart}>
            <GitBranch size={16} /> Run Again
          </button>
        </div>
      )}
    </div>
  )
}
