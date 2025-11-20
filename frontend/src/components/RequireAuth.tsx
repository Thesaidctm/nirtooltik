import { Navigate, useLocation } from 'react-router-dom'

import { getAuthToken } from '../api/client'

interface RequireAuthProps {
  children: React.ReactElement
}

const RequireAuth = ({ children }: RequireAuthProps) => {
  const token = getAuthToken()
  const location = useLocation()

  if (!token) {
    return <Navigate to="/login" state={{ from: location }} replace />
  }

  return children
}

export default RequireAuth
