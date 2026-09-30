import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Database, GitMerge, Cpu, FileOutput, ArrowRight,
  Upload, BarChart3, Layers, Zap, TrendingUp, Users
} from 'lucide-react'
import { getBackendHealth, getResults } from '../services/api'

const WORKFLOW_STEPS = [
  { icon: Database, label: 'Data Sources', color: 'var(--blue-400)', bg: 'rgba(59,130,246,0.1)' },
  { icon: Layers, label: 'Preprocessing', color: 'var(--purple-400)', bg: 'rgba(168,85,247,0.1)' },
  { icon: Users, label: 'Candidate Gen', color: 'var(--cyan-400)', bg: 'rgba(34,211,238,0.1)' },
  { icon: TrendingUp, label: 'Similarity Analysis', color: 'var(--yellow-400)', bg: 'rgba(250,204,21,0.1)' },
  { icon: Cpu, label: 'ML Matching', color: 'var(--purple-400)', bg: 'rgba(168,85,247,0.1)' },
  { icon: FileOutput, label: 'Results', color: 'var(--green-400)', bg: 'rgba(74,222,128,0.1)' },
]

export default function Dashboard() {
  const [health, setHealth] = useState(null)
  const [stats, setStats] = useState(null)

  useEffect(() => {
    getBackendHealth().then(setHealth).catch(() => setHealth({ status: 'error' }))
    getResults()
      .then(d => setStats(d.statistics))
      .catch(() => {}) // no results yet is fine
  }, [])

  const s = stats || {}

  return (
    <div>
      {/* Ambient orbs */}
      <div className="orb orb-1" />
      <div className="orb orb-2" />

      {/* Hero */}
      <div style={{ textAlign: 'center', padding: '3rem 1rem 2rem' }}>
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: '0.5rem',
          padding: '0.375rem 1rem', borderRadius: '20px',
          background: 'rgba(59,130,246,0.1)', border: '1px solid rgba(59,130,246,0.25)',
          fontSize: '0.8rem', fontWeight: 600, color: 'var(--blue-400)', marginBottom: '1.25rem'
        }}>
          <Zap size={13} />
          Intelligent Business Entity Resolution
        </div>
        <h1 style={{
          fontSize: 'clamp(2rem, 5vw, 3.25rem)',
          fontWeight: 900, letterSpacing: '-0.04em',
          color: 'var(--text-primary)', lineHeight: 1.1, marginBottom: '1rem'
        }}>
          Resolve Business Entities <br />
          <span className="text-gradient">with AI</span>
        </h1>
        <p style={{
          fontSize: '1rem', color: 'var(--text-secondary)',
          maxWidth: 560, margin: '0 auto 2rem', lineHeight: 1.7
        }}>
          Identify matching businesses across noisy and inconsistent data sources
          using intelligent entity resolution — powered by a trained ML model with
          99.5% precision.
        </p>
        <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
          <Link to="/upload" className="btn btn-primary btn-lg">
            <Upload size={18} />
            Start New Matching
          </Link>
          <Link to="/results" className="btn btn-secondary btn-lg">
            <BarChart3 size={18} />
            View Results
          </Link>
        </div>
      </div>

      {/* Stats */}
      <div className="stat-grid" style={{ marginTop: '2rem' }}>
        {[
          {
            label: 'Source 1 Records', value: s.source1_entities ?? '—',
            icon: Database, color: 'var(--blue-400)', bg: 'rgba(59,130,246,0.12)'
          },
          {
            label: 'Matched Entities', value: s.matched_entities ?? '—',
            icon: GitMerge, color: 'var(--green-400)', bg: 'rgba(74,222,128,0.12)'
          },
          {
            label: 'Unmatched Entities', value: s.unmatched_entities ?? '—',
            icon: Users, color: 'var(--yellow-400)', bg: 'rgba(250,204,21,0.12)'
          },
          {
            label: 'Match Rate', value: s.match_rate != null ? `${s.match_rate}%` : '—',
            icon: TrendingUp, color: 'var(--purple-400)', bg: 'rgba(168,85,247,0.12)'
          },
        ].map(({ label, value, icon: Icon, color, bg }) => (
          <div key={label} className="stat-card">
            <div className="stat-icon" style={{ background: bg }}>
              <Icon size={20} color={color} />
            </div>
            <div className="stat-label">{label}</div>
            <div className="stat-value">{typeof value === 'number' ? value.toLocaleString() : value}</div>
          </div>
        ))}
      </div>

      {/* Pipeline */}
      <div className="card" style={{ marginTop: '1.5rem' }}>
        <div style={{ marginBottom: '1.25rem' }}>
          <div className="section-label">Processing Pipeline</div>
          <h2 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            End-to-End Entity Resolution Workflow
          </h2>
        </div>
        <div className="pipeline-flow">
          {WORKFLOW_STEPS.map((step, i) => (
            <div key={step.label} style={{ display: 'contents' }}>
              <div className="pipeline-step">
                <div className="pipeline-step-icon" style={{ background: step.bg }}>
                  <step.icon size={18} color={step.color} />
                </div>
                <div className="pipeline-step-label">{step.label}</div>
              </div>
              {i < WORKFLOW_STEPS.length - 1 && (
                <div className="pipeline-arrow">
                  <ArrowRight size={16} color="var(--text-muted)" />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* ML Status card */}
      {health && (
        <div className="card" style={{ marginTop: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
            <div>
              <div className="section-label">ML Model Status</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <div style={{
                  width: 10, height: 10, borderRadius: '50%',
                  background: health.ml_model === 'ready' ? 'var(--green-400)' : 'var(--red-400)',
                  animation: 'pulse 2s infinite'
                }} />
                <span style={{
                  fontSize: '0.95rem', fontWeight: 700,
                  color: health.ml_model === 'ready' ? 'var(--green-400)' : 'var(--red-400)'
                }}>
                  {health.ml_model === 'ready' ? 'All ML Artifacts Loaded & Ready' : 'ML Model Not Available'}
                </span>
              </div>
              {health.ml_model_details && (
                <div style={{ marginTop: '0.5rem', display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
                  {health.ml_model_details.model_name && (
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                      Model: <span style={{ color: 'var(--text-secondary)' }}>{health.ml_model_details.model_name}</span>
                    </div>
                  )}
                  {health.ml_model_details.threshold && (
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                      Threshold: <span style={{ color: 'var(--text-secondary)' }}>{health.ml_model_details.threshold}</span>
                    </div>
                  )}
                  {health.ml_model_details.feature_count && (
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                      Features: <span style={{ color: 'var(--text-secondary)' }}>{health.ml_model_details.feature_count}</span>
                    </div>
                  )}
                </div>
              )}
            </div>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <Link to="/upload" className="btn btn-secondary btn-sm">
                <Upload size={14} /> Upload Data
              </Link>
              <Link to="/results" className="btn btn-ghost btn-sm">
                <BarChart3 size={14} /> Results
              </Link>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
