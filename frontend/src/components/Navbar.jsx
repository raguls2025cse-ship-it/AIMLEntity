import { useState, useEffect } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { Zap, Upload, GitBranch, BarChart3, Info, CheckCircle2, XCircle, Loader2 } from 'lucide-react'
import { getBackendHealth } from '../services/api'

const NAV_ITEMS = [
  { to: '/', label: 'Dashboard', icon: BarChart3 },
  { to: '/upload', label: 'Upload', icon: Upload },
  { to: '/matching', label: 'Matching', icon: GitBranch },
  { to: '/results', label: 'Results', icon: BarChart3 },
  { to: '/about', label: 'About', icon: Info },
]

export default function Navbar() {
  const [health, setHealth] = useState(null) // null = checking
  const location = useLocation()

  useEffect(() => {
    let cancelled = false
    async function check() {
      try {
        const data = await getBackendHealth()
        if (!cancelled) setHealth(data)
      } catch {
        if (!cancelled) setHealth({ ml_model: 'not_ready', status: 'error' })
      }
    }
    check()
    const interval = setInterval(check, 30000)
    return () => { cancelled = true; clearInterval(interval) }
  }, [])

  const mlReady = health?.ml_model === 'ready'
  const checking = health === null

  return (
    <nav className="navbar">
      <NavLink to="/" className="navbar-brand">
        <div className="brand-icon">
          <Zap size={18} color="white" />
        </div>
        <span className="brand-name">EntitySync AI</span>
      </NavLink>

      <div className="nav-links">
        {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
            end={to === '/'}
          >
            <Icon size={15} />
            <span>{label}</span>
          </NavLink>
        ))}
      </div>

      <div
        className={`health-badge ${checking ? 'checking' : mlReady ? 'ready' : 'not-ready'}`}
        title={health?.ml_model_details?.message || 'Checking ML model…'}
      >
        <div className={`pulse-dot ${checking ? 'yellow' : mlReady ? 'green' : 'red'}`} />
        {checking ? 'Checking…' : mlReady ? 'ML Ready' : 'ML Offline'}
      </div>
    </nav>
  )
}
