import { CheckCircle2, Circle, Loader2, XCircle } from 'lucide-react'

const PIPELINE_STAGES = [
  'Initializing ML model',
  'Loading source datasets',
  'Preprocessing records',
  'Generating candidate pairs (blocking)',
  'Extracting features',
  'Running ML model inference',
  'Applying threshold and generating results',
  'Generating output files',
  'Completed',
]

function matchStage(currentStage) {
  const cur = (currentStage || '').toLowerCase()
  for (let i = 0; i < PIPELINE_STAGES.length; i++) {
    if (cur.includes(PIPELINE_STAGES[i].toLowerCase().slice(0, 12))) return i
    // broader match
    if (PIPELINE_STAGES[i].toLowerCase().includes(cur.slice(0, 10))) return i
  }
  if (cur.includes('complet')) return PIPELINE_STAGES.length - 1
  if (cur.includes('fail')) return -1
  return 0
}

export default function ProgressPanel({ jobStatus }) {
  if (!jobStatus) return null

  const { status, stage, progress, statistics, error } = jobStatus
  const failed = status === 'failed'
  const completed = status === 'completed'

  const activeIdx = completed ? PIPELINE_STAGES.length
    : failed ? -1
    : matchStage(stage)

  return (
    <div className="progress-container">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h3 style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-primary)' }}>
          Pipeline Progress
        </h3>
        {!completed && !failed && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem', color: 'var(--blue-400)' }}>
            <div className="spinner" style={{ width: 14, height: 14 }} />
            Processing…
          </div>
        )}
      </div>

      {/* Progress bar */}
      <div className="progress-bar-wrap" style={{ marginBottom: '1rem' }}>
        <div className="progress-bar-fill" style={{ width: `${progress || 0}%` }} />
      </div>
      <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
        {progress || 0}% — {stage || 'Waiting…'}
      </p>

      <div className="progress-stages">
        {PIPELINE_STAGES.map((s, i) => {
          let cls = 'pending'
          let Icon = Circle
          if (completed || i < activeIdx) { cls = 'done'; Icon = CheckCircle2 }
          else if (i === activeIdx && !failed) { cls = 'active'; Icon = Loader2 }
          else if (failed && i === activeIdx) { cls = 'failed'; Icon = XCircle }

          return (
            <div key={s} className={`progress-stage ${cls}`}>
              <Icon size={15} style={{ flexShrink: 0, ...(cls === 'active' ? { animation: 'spin 1s linear infinite' } : {}) }} />
              {s}
            </div>
          )
        })}
      </div>

      {error && (
        <div className="alert alert-error" style={{ marginTop: '1rem' }}>
          <XCircle size={15} />
          <div>
            <strong>Error:</strong> {error}
          </div>
        </div>
      )}

      {completed && statistics && (
        <div className="alert alert-success" style={{ marginTop: '1rem', flexDirection: 'column', gap: '0.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 700 }}>
            <CheckCircle2 size={15} /> Entity Resolution Completed!
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.5rem', fontSize: '0.8rem', marginTop: '0.25rem' }}>
            <div><span style={{ color: 'var(--text-muted)' }}>S1 Entities: </span><strong>{statistics.source1_entities?.toLocaleString()}</strong></div>
            <div><span style={{ color: 'var(--text-muted)' }}>Matched: </span><strong>{statistics.matched_entities?.toLocaleString()}</strong></div>
            <div><span style={{ color: 'var(--text-muted)' }}>Match Rate: </span><strong>{statistics.match_rate}%</strong></div>
          </div>
        </div>
      )}
    </div>
  )
}
