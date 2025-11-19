import { NavLink } from 'react-router-dom'

const navItems = [
  { label: 'Dashboard', to: '/' },
  { label: 'Datasets', to: '/datasets' },
  { label: 'Calibrações', to: '/runs' },
  { label: 'Modelos', to: '/models' },
  { label: 'Transferência de Calibração', to: '/transfers' },
  { label: 'Predição Online', to: '/predict' },
]

const Sidebar = () => {
  return (
    <aside className="fixed inset-y-0 left-0 w-64 bg-slate-900 text-slate-100 shadow-xl">
      <div className="p-6 text-2xl font-bold tracking-tight">NIRCLS</div>
      <nav className="mt-4 flex flex-col space-y-2 px-4">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              `rounded-xl px-4 py-3 text-sm font-medium transition-colors duration-150 hover:bg-slate-800/80 hover:text-white ${
                isActive ? 'bg-slate-800 text-white shadow-inner' : 'text-slate-200'
              }`
            }
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  )
}

export default Sidebar
