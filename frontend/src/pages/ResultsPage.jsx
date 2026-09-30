import { useState, useEffect, useMemo } from 'react'
import { Search, Download, Eye, RefreshCw, AlertCircle, Database } from 'lucide-react'
import { getResults, downloadMatchingResults, downloadCandidatePairs } from '../services/api'
import MatchDetailModal from '../components/MatchDetailModal'

function ConfidenceCell({ value }) {
  if (value === null || value === undefined) {
    return <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>—</span>
  }
  const pct = Math.round(value * 100)
  const color = pct >= 90 ? 'var(--green-400)' : pct >= 70 ? 'var(--blue-400)' : 'var(--yellow-400)'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', minWidth: 100 }}>
      <div style={{ flex: 1, height: 4, background: 'rgba(255,255,255,0.07)', borderRadius: 2, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: 2 }} />
      </div>
      <span style={{ fontSize: '0.72rem', fontWeight: 600, color, fontFamily: 'JetBrains Mono, monospace', minWidth: 32 }}>
        {pct}%
      </span>
    </div>
  )
}

function maxConfidence(matched_details) {
  if (!matched_details?.length) return null
  const vals = matched_details.map(m => m.confidence).filter(v => v !== null && v !== undefined)
  return vals.length ? Math.max(...vals) : null
}

export default function ResultsPage() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState('all')
  const [selectedRow, setSelectedRow] = useState(null)
  const [page, setPage] = useState(1)
  const PAGE_SIZE = 50

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const d = await getResults()
      setData(d)
    } catch (e) {
      setError(e.message || 'Failed to load results')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const filtered = useMemo(() => {
    if (!data?.rows) return []
    let rows = data.rows
    if (filter === 'matched') rows = rows.filter(r => r.status === 'matched')
    if (filter === 'unmatched') rows = rows.filter(r => r.status === 'unmatched')
    if (search.trim()) {
      const q = search.trim().toLowerCase()
      rows = rows.filter(r =>
        r.source1_entity_id?.toLowerCase().includes(q) ||
        r.business_name?.toLowerCase().includes(q)
      )
    }
    return rows
  }, [data, filter, search])

  const paged = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)
  const totalPages = Math.ceil(filtered.length / PAGE_SIZE)

  if (loading) return (
    <div style={{ textAlign: 'center', padding: '4rem 2rem', color: 'var(--text-muted)' }}>
      <div className="spinner" style={{ margin: '0 auto 1rem', width: 32, height: 32, borderWidth: 3 }} />
      Loading results…
    </div>
  )

  if (error) return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Results</h1>
      </div>
      <div className="alert alert-error" style={{ marginBottom: '1rem' }}>
        <AlertCircle size={16} />
        <div>
          <strong>Results not available:</strong> {error}
          <div style={{ marginTop: '0.375rem', fontSize: '0.8rem' }}>
            Run the entity resolution pipeline on the{' '}
            <a href="/matching" style={{ color: 'inherit', textDecoration: 'underline' }}>Matching page</a> first.
          </div>
        </div>
      </div>
    </div>
  )

  const s = data?.statistics || {}

  return (
    <div>
      <div className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <h1 className="page-title">Results</h1>
            <p className="page-subtitle">Entity resolution results from the trained ML model.</p>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            <button className="btn btn-ghost btn-sm" onClick={load}>
              <RefreshCw size={14} /> Refresh
            </button>
            <button className="btn btn-secondary btn-sm" onClick={downloadMatchingResults} id="download-matching-btn">
              <Download size={14} /> matching_results.tsv
            </button>
            <button className="btn btn-ghost btn-sm" onClick={downloadCandidatePairs} id="download-candidates-btn">
              <Download size={14} /> candidate_pairs.tsv
            </button>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="stat-grid" style={{ marginBottom: '1.5rem' }}>
        {[
          { label: 'Source 1 Entities', value: s.source1_entities, color: 'var(--blue-400)', bg: 'rgba(59,130,246,0.1)' },
          { label: 'Matched', value: s.matched_entities, color: 'var(--green-400)', bg: 'rgba(74,222,128,0.1)' },
          { label: 'Unmatched', value: s.unmatched_entities, color: 'var(--yellow-400)', bg: 'rgba(250,204,21,0.1)' },
          { label: 'Match Rate', value: s.match_rate != null ? `${s.match_rate}%` : '—', color: 'var(--purple-400)', bg: 'rgba(168,85,247,0.1)' },
        ].map(({ label, value, color, bg }) => (
          <div key={label} className="stat-card">
            <div className="stat-icon" style={{ background: bg }}>
              <Database size={18} color={color} />
            </div>
            <div className="stat-label">{label}</div>
            <div className="stat-value">{typeof value === 'number' ? value.toLocaleString() : value ?? '—'}</div>
          </div>
        ))}
      </div>

      {/* Search + filter */}
      <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap' }}>
        <div className="search-bar">
          <Search size={15} className="search-icon" />
          <input
            className="search-input"
            placeholder="Search entity ID or business name…"
            value={search}
            onChange={e => { setSearch(e.target.value); setPage(1) }}
          />
        </div>
        <div className="filter-group">
          {['all', 'matched', 'unmatched'].map(f => (
            <button
              key={f}
              className={`filter-btn ${filter === f ? 'active' : ''}`}
              onClick={() => { setFilter(f); setPage(1) }}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
          {filtered.length.toLocaleString()} records
        </span>
      </div>

      {/* Table */}
      <div className="data-table-wrap" style={{ marginBottom: '1rem' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Source 1 ID</th>
              <th>Business Name</th>
              <th>Matched Entity IDs</th>
              <th>Confidence</th>
              <th>Status</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {paged.length === 0 ? (
              <tr>
                <td colSpan={6} style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
                  No records found.
                </td>
              </tr>
            ) : paged.map(row => (
              <tr key={row.source1_entity_id}>
                <td><span className="entity-id">{row.source1_entity_id}</span></td>
                <td style={{ color: 'var(--text-primary)', fontWeight: 500, maxWidth: 200 }}>
                  {row.business_name || <em style={{ color: 'var(--text-muted)' }}>—</em>}
                </td>
                <td style={{ maxWidth: 240 }}>
                  {row.matched_entity_ids ? (
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem' }}>
                      {row.matched_entity_ids.split(',').slice(0, 5).map(id => {
                        const trimmed = id.trim()
                        return (
                          <span key={trimmed} className={`badge ${trimmed.startsWith('S2-') ? 'badge-s2' : 'badge-s3'}`}>
                            {trimmed}
                          </span>
                        )
                      })}
                      {row.matched_entity_ids.split(',').length > 5 && (
                        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                          +{row.matched_entity_ids.split(',').length - 5} more
                        </span>
                      )}
                    </div>
                  ) : <em style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>No match</em>}
                </td>
                <td><ConfidenceCell value={maxConfidence(row.matched_details)} /></td>
                <td>
                  <span className={`badge ${row.status === 'matched' ? 'badge-matched' : 'badge-unmatched'}`}>
                    {row.status}
                  </span>
                </td>
                <td>
                  <button
                    className="btn btn-ghost btn-sm"
                    onClick={() => setSelectedRow(row)}
                    title="View match details"
                  >
                    <Eye size={13} />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', justifyContent: 'center' }}>
          <button className="btn btn-ghost btn-sm" disabled={page === 1} onClick={() => setPage(p => p - 1)}>
            Prev
          </button>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Page {page} of {totalPages}
          </span>
          <button className="btn btn-ghost btn-sm" disabled={page === totalPages} onClick={() => setPage(p => p + 1)}>
            Next
          </button>
        </div>
      )}

      {selectedRow && (
        <MatchDetailModal row={selectedRow} onClose={() => setSelectedRow(null)} />
      )}
    </div>
  )
}
