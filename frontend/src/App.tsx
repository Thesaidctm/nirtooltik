import { Navigate, Route, Routes } from 'react-router-dom'

import AppLayout from './components/AppLayout'
import Dashboard from './pages/Dashboard'
import Datasets from './pages/Datasets'
import ModelDetails from './pages/ModelDetails'
import Models from './pages/Models'
import Predict from './pages/Predict'
import RunDetails from './pages/RunDetails'
import Runs from './pages/Runs'
import TransferDetails from './pages/TransferDetails'
import Transfers from './pages/Transfers'

const App = () => {
  return (
    <Routes>
      <Route path="/" element={<AppLayout />}>
        <Route index element={<Navigate to="/datasets" replace />} />
        <Route path="dashboard" element={<Dashboard />} />
        <Route path="datasets" element={<Datasets />} />
        <Route path="runs" element={<Runs />} />
        <Route path="runs/:runId" element={<RunDetails />} />
        <Route path="models" element={<Models />} />
        <Route path="models/:modelId" element={<ModelDetails />} />
        <Route path="transfers" element={<Transfers />} />
        <Route path="transfers/:transferId" element={<TransferDetails />} />
        <Route path="predict" element={<Predict />} />
      </Route>
    </Routes>
  )
}

export default App
