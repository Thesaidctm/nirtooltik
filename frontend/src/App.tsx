import { Route, Routes } from 'react-router-dom'

import AppLayout from './components/AppLayout'
import Dashboard from './pages/Dashboard'
import Datasets from './pages/Datasets'
import Models from './pages/Models'
import Predict from './pages/Predict'
import Runs from './pages/Runs'
import Transfers from './pages/Transfers'

const App = () => {
  return (
    <Routes>
      <Route path="/" element={<AppLayout />}>
        <Route index element={<Dashboard />} />
        <Route path="datasets" element={<Datasets />} />
        <Route path="runs" element={<Runs />} />
        <Route path="models" element={<Models />} />
        <Route path="transfers" element={<Transfers />} />
        <Route path="predict" element={<Predict />} />
      </Route>
    </Routes>
  )
}

export default App
