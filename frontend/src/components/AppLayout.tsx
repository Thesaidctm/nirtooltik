import { Outlet } from 'react-router-dom'

import Sidebar from './Sidebar'
import Topbar from './Topbar'

const AppLayout = () => {
  return (
    <div className="min-h-screen bg-slate-50">
      <Sidebar />
      <div className="ml-64 flex min-h-screen flex-col">
        <Topbar />
        <main className="flex-1 p-6">
          <div className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-100">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  )
}

export default AppLayout
