import { X, Building2, MapPin, Globe, Activity } from 'lucide-react'

function ConfidenceBar({ value }) {
  if (value === null || value === undefined) {
    return <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>N/A</span>
  }
  const pct = Math.round(value * 100)
  const color = pct >= 90 ? 'var(--green-400)' : pct >= 70 ? 'var(--blue-400)' : 'var(--yellow-400)'
  return (
    <div className="confidence-bar">
      <div className="conf-track">
        <div className="conf-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="conf-label" style={{ color }}>{pct}%</span>
    </div>
  )
}

function EntityBlock({ title, data, color }) {
  return (
    <div style={{
      padding: '1rem', borderRadius: '12px',
      background: 'rgba(10, 22, 40, 0.5)',
      border: `1px solid ${color}30`
    }}>
      <div style={{ fontSize: '0.7rem', fontWeight: 700, textTransform: 'uppercase',
                    letterSpacing: '0.08em', color, marginBottom: '0.75rem' }}>{title}</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'flex-start' }}>
          <Building2 size={14} color="var(--text-muted)" style={{ marginTop: 2, flexShrink: 0 }} />
          <div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Business Name</div>
            <div style={{ fontSize: '0.875rem', color: 'var(--text-primary)', fontWeight: 500 }}>
              {data?.business_name || <em style={{ color: 'var(--text-muted)' }}>—</em>}
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'flex-start' }}>
          <MapPin size={14} color="var(--text-muted)" style={{ marginTop: 2, flexShrink: 0 }} />
          <div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Address</div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              {data?.business_address || <em style={{ color: 'var(--text-muted)' }}>—</em>}
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'flex-start' }}>
          <Globe size={14} color="var(--text-muted)" style={{ marginTop: 2, flexShrink: 0 }} />
          <div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Country</div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              {data?.country || <em style={{ color: 'var(--text-muted)' }}>—</em>}
            </div>
          </div>
        </div>
        {data?.confidence !== undefined && (
          <div style={{ marginTop: '0.25rem' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '0.3rem' }}>
              ML Confidence
            </div>
            <ConfidenceBar value={data.confidence} />
          </div>
        )}
      </div>
    </div>
  )
}

export default function MatchDetailModal({ row, onClose }) {
  if (!row) return null

  const s1Data = {
    business_name: row.business_name,
    business_address: row.business_address,
    country: row.country,
  }

  const s2Matches = (row.matched_details || []).filter(m => m.source === 'S2')
  const s3Matches = (row.matched_details || []).filter(m => m.source === 'S3')

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.25rem' }}>
          <div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '0.3rem' }}>
              MATCH DETAILS
            </div>
            <h2 style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              {row.business_name || row.source1_entity_id}
            </h2>
            <div className="mono" style={{ fontSize: '0.75rem', color: 'var(--blue-400)', marginTop: '0.2rem' }}>
              {row.source1_entity_id}
            </div>
          </div>
          <button className="btn btn-ghost btn-sm" onClick={onClose}>
            <X size={16} />
          </button>
        </div>

        {/* Source 1 */}
        <div style={{ marginBottom: '1rem' }}>
          <div className="section-label">Reference Record (Source 1)</div>
          <EntityBlock title="Source 1" data={s1Data} color="var(--blue-400)" />
        </div>

        {/* Matched records */}
        {row.matched_details?.length > 0 ? (
          <>
            {s2Matches.length > 0 && (
              <div style={{ marginBottom: '1rem' }}>
                <div className="section-label">Matched — Source 2</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {s2Matches.map(m => (
                    <EntityBlock key={m.entity_id} title={`S2 · ${m.entity_id}`} data={m} color="var(--purple-400)" />
                  ))}
                </div>
              </div>
            )}
            {s3Matches.length > 0 && (
              <div style={{ marginBottom: '1rem' }}>
                <div className="section-label">Matched — Source 3</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {s3Matches.map(m => (
                    <EntityBlock key={m.entity_id} title={`S3 · ${m.entity_id}`} data={m} color="var(--cyan-400)" />
                  ))}
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="alert alert-warning" style={{ marginTop: '0.5rem' }}>
            <Activity size={15} />
            No matching entities found above the confidence threshold (0.70).
          </div>
        )}
      </div>
    </div>
  )
}
