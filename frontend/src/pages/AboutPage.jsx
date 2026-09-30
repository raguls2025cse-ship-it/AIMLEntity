import { Brain, Code2, Shield, ArrowRight, Database, Cpu, GitMerge, FileOutput } from 'lucide-react'

const ARCH_STEPS = [
  { icon: Database, label: 'Source 1 / 2 / 3', desc: 'TSV business records', color: 'var(--blue-400)', bg: 'rgba(59,130,246,0.1)' },
  { icon: Code2, label: 'Preprocessing', desc: 'Name, address, country normalization', color: 'var(--cyan-400)', bg: 'rgba(34,211,238,0.1)' },
  { icon: GitMerge, label: 'Candidate Generation', desc: 'Multi-strategy blocking (99.75% recall)', color: 'var(--purple-400)', bg: 'rgba(168,85,247,0.1)' },
  { icon: Brain, label: 'Feature Extraction', desc: '37 similarity signals', color: 'var(--yellow-400)', bg: 'rgba(250,204,21,0.1)' },
  { icon: Cpu, label: 'Trained ML Model', desc: 'HistGradientBoosting · Threshold 0.70', color: 'var(--green-400)', bg: 'rgba(74,222,128,0.1)' },
  { icon: FileOutput, label: 'Results', desc: 'matching_results.tsv + candidate_pairs.tsv', color: 'var(--blue-400)', bg: 'rgba(59,130,246,0.1)' },
]

const TECH_STACK = [
  { cat: 'Frontend', items: ['React', 'Vite', 'React Router', 'Lucide React', 'Vanilla CSS'] },
  { cat: 'Backend', items: ['Python', 'FastAPI', 'Uvicorn', 'Pandas', 'python-multipart'] },
  { cat: 'Machine Learning', items: ['scikit-learn', 'RapidFuzz', 'joblib', 'NumPy', 'SciPy'] },
  { cat: 'Data Processing', items: ['TSV Parsing', 'TF-IDF Vectorizers', 'Fuzzy Matching', 'Blocking Indices'] },
]

export default function AboutPage() {
  return (
    <div>
      <div className="page-header">
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: '0.5rem',
          padding: '0.375rem 1rem', borderRadius: '20px',
          background: 'rgba(59,130,246,0.1)', border: '1px solid rgba(59,130,246,0.25)',
          fontSize: '0.8rem', fontWeight: 600, color: 'var(--blue-400)', marginBottom: '1rem'
        }}>
          <Shield size={13} /> AIML Hackathon — Business Entity Resolution
        </div>
        <h1 className="page-title">About EntitySync AI</h1>
        <p className="page-subtitle" style={{ maxWidth: 600 }}>
          EntitySync AI identifies records from independent business data sources that represent
          the same real-world business — powered entirely by an on-premise trained ML model.
          No external APIs. No business lookup services. No internet-based enrichment.
        </p>
      </div>

      {/* What it does */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div className="section-label">Problem Statement</div>
        <h2 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '0.75rem' }}>
          Business Entity Resolution
        </h2>
        <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: 1.75, marginBottom: '1rem' }}>
          Large organizations receive business data from multiple independent sources. The same
          company might appear as <em>"Acme Technologies Inc."</em> in Source 1,{' '}
          <em>"ACME TECHNOLOGIES INCORPORATED"</em> in Source 2, and{' '}
          <em>"Acme Tech Intl"</em> in Source 3. EntitySync AI resolves these to a single
          canonical reference entity using machine learning — not simple string matching.
        </p>
        <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
          {[
            { label: 'Precision', value: '99.53%' },
            { label: 'Recall', value: '97.64%' },
            { label: 'Macro F0.5', value: '98.98%' },
            { label: 'Blocking Recall', value: '99.75%' },
          ].map(({ label, value }) => (
            <div key={label}>
              <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-primary)' }}>{value}</div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Architecture */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div className="section-label">System Architecture</div>
        <h2 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '1.25rem' }}>
          End-to-End Pipeline
        </h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.625rem' }}>
          {ARCH_STEPS.map((step, i) => (
            <div key={step.label} style={{ display: 'flex', flexDirection: 'column', gap: '0' }}>
              <div style={{
                display: 'flex', alignItems: 'center', gap: '1rem', padding: '0.875rem 1rem',
                borderRadius: '10px', background: step.bg,
                border: `1px solid ${step.color}25`,
              }}>
                <div style={{
                  width: 36, height: 36, borderRadius: 9, display: 'flex', alignItems: 'center',
                  justifyContent: 'center', background: `${step.color}20`, flexShrink: 0
                }}>
                  <step.icon size={17} color={step.color} />
                </div>
                <div>
                  <div style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)' }}>{step.label}</div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{step.desc}</div>
                </div>
              </div>
              {i < ARCH_STEPS.length - 1 && (
                <div style={{ display: 'flex', justifyContent: 'center', padding: '0.15rem 0' }}>
                  <ArrowRight size={14} color="var(--text-muted)" style={{ transform: 'rotate(90deg)' }} />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* ML Features */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div className="section-label">ML Feature Engineering</div>
        <h2 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '1rem' }}>
          37 Similarity Signals
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.75rem' }}>
          {[
            { cat: 'Business Name (12)', items: ['Ratio, WRatio, QRatio', 'Token Sort/Set/Partial', 'Length diff & ratio', 'Token Jaccard', 'TF-IDF cosine', 'Prefix match'] },
            { cat: 'Address (12)', items: ['Ratio, WRatio, QRatio', 'Token Sort/Set/Partial', 'Length diff & ratio', 'Token Jaccard', 'TF-IDF cosine', 'Numeric overlap'] },
            { cat: 'Cross-field (7)', items: ['Country exact match', 'Country normalized', 'Country conflict', 'Combined TF-IDF', 'Weighted sim', 'Shared tokens', 'Source indicator'] },
            { cat: 'Missing Flags (6)', items: ['S1 name/addr/country missing', 'Cand name/addr/country missing'] },
          ].map(({ cat, items }) => (
            <div key={cat} style={{ padding: '0.875rem', borderRadius: '10px', background: 'rgba(10,22,40,0.5)', border: '1px solid var(--border)' }}>
              <div style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--blue-400)', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{cat}</div>
              {items.map(item => (
                <div key={item} style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', padding: '0.1rem 0' }}>· {item}</div>
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* Technology Stack */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div className="section-label">Technology Stack</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
          {TECH_STACK.map(({ cat, items }) => (
            <div key={cat} style={{ padding: '0.875rem', borderRadius: '10px', background: 'rgba(10,22,40,0.5)', border: '1px solid var(--border)' }}>
              <div style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--purple-400)', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{cat}</div>
              {items.map(item => (
                <div key={item} style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', padding: '0.1rem 0' }}>· {item}</div>
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* Security notice */}
      <div className="alert alert-info">
        <Shield size={16} />
        <div>
          <strong>Data Privacy Notice:</strong> All processing happens on-premise using only the datasets you upload.
          EntitySync AI does not use external APIs, business databases, geocoding services, or any internet-based enrichment.
          Your data never leaves your server.
        </div>
      </div>
    </div>
  )
}
