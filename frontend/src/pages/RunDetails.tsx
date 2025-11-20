import { useParams } from 'react-router-dom'
import { API_BASE_URL } from '../api/client'
import RunDetailsPage from '../components/runs/RunDetailsPage'

const RunDetails = () => {
  const { runId } = useParams<{ runId: string }>()

  if (!runId) {
    return <div className="py-12 text-center text-slate-500">Nenhum run selecionado.</div>
  }

  return <RunDetailsPage runId={runId} apiBaseUrl={API_BASE_URL} />
}

export default RunDetails
