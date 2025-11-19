import { NavLink } from 'react-router-dom'

const navItems = [
  { label: 'Dashboard', to: '/dashboard' },
  { label: 'Datasets', to: '/datasets' },
  { label: 'Runs', to: '/runs' },
  { label: 'Models', to: '/models' },
  { label: 'Transfers', to: '/transfers' },
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
            end={item.to === '/dashboard'}
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
