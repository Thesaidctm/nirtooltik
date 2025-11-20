import { useState } from 'react'
import type { Location } from 'react-router-dom'
import { useLocation, useNavigate } from 'react-router-dom'

import { API_BASE_URL, setAuthToken } from '../api/client'

const LoginPage = () => {
  const [username, setUsername] = useState('admin')
  const [password, setPassword] = useState('admin')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const navigate = useNavigate()
  const location = useLocation()
  const redirectTo = (location.state as { from?: Location })?.from?.pathname ?? '/datasets'

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    setError(null)
    setLoading(true)

    try {
      const body = new URLSearchParams()
      body.append('username', username)
      body.append('password', password)
      body.append('grant_type', 'password')

      const response = await fetch(`${API_BASE_URL}/auth/token`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body,
      })

      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || 'Falha no login')
      }

      const data = await response.json()
      setAuthToken(data.access_token)
      navigate(redirectTo)
    } catch (err) {
      console.error(err)
      setError('Falha no login. Verifique usuário e senha.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-100 p-4">
      <div className="w-full max-w-md rounded-2xl bg-white p-8 shadow-xl">
        <h1 className="mb-2 text-center text-2xl font-semibold text-slate-900">NIRCLS</h1>
        <p className="mb-6 text-center text-sm text-slate-500">Entre para acessar o painel</p>

        {error ? (
          <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div>
        ) : null}

        <form className="space-y-4" onSubmit={handleSubmit}>
          <div>
            <label className="mb-1 block text-sm font-medium text-slate-700" htmlFor="username">
              Usuário
            </label>
            <input
              id="username"
              type="text"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              placeholder="admin"
              autoComplete="username"
            />
          </div>

          <div>
            <label className="mb-1 block text-sm font-medium text-slate-700" htmlFor="password">
              Senha
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              placeholder="admin"
              autoComplete="current-password"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700 disabled:opacity-60"
          >
            {loading ? 'Entrando...' : 'Entrar'}
          </button>
        </form>
      </div>
    </div>
  )
}

export default LoginPage
