import { useParams } from 'react-router-dom'
import RunDetailsPage from '../components/runs/RunDetailsPage'

const API_BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const RunDetails = () => {
  const { runId } = useParams<{ runId: string }>()

  if (!runId) {
    return <div className="py-12 text-center text-slate-500">Nenhum run selecionado.</div>
  }

  return <RunDetailsPage runId={runId} apiBaseUrl={API_BASE_URL} />
}

export default RunDetails
