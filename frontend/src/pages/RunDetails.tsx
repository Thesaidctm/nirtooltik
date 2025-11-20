import { useParams } from 'react-router-dom'
import RunDetailsPage from '../components/runs/RunDetailsPage'

const RunDetails = () => {
  const { runId } = useParams<{ runId: string }>()

  if (!runId) {
    return <div className="py-12 text-center text-slate-500">Nenhum run selecionado.</div>
  }

  return <RunDetailsPage runId={runId} />
}

export default RunDetails
