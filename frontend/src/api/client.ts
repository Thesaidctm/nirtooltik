const buildBaseUrl = () => {
  const fromEnv = import.meta.env?.VITE_API_URL ?? import.meta.env?.VITE_API_BASE_URL
  return fromEnv && typeof fromEnv === 'string' && fromEnv.length > 0 ? fromEnv : 'http://localhost:8000'
}

export const API_BASE_URL = buildBaseUrl()
const TOKEN_KEY = 'nircls_token'

export const getAuthToken = () => localStorage.getItem(TOKEN_KEY)
export const setAuthToken = (token: string) => localStorage.setItem(TOKEN_KEY, token)
export const clearAuthToken = () => localStorage.removeItem(TOKEN_KEY)

const buildUrl = (path: string) => {
  if (path.startsWith('http://') || path.startsWith('https://')) {
    return path
  }
  if (path.startsWith('/')) {
    return `${API_BASE_URL}${path}`
  }
  return `${API_BASE_URL}/${path}`
}

const parseErrorMessage = async (response: Response) => {
  try {
    const data = await response.json()
    if (typeof data === 'string') return data
    if (data?.detail) {
      if (Array.isArray(data.detail)) {
        return data.detail
          .map((detail: { msg?: string } | string) =>
            typeof detail === 'string' ? detail : detail.msg ?? JSON.stringify(detail),
          )
          .join(', ')
      }
      return data.detail
    }
    if (data?.message) return data.message
    return JSON.stringify(data)
  } catch (error) {
    const text = await response.text()
    return text || 'Unknown error'
  }
}

const request = async <T>(method: string, path: string, body?: BodyInit | Record<string, unknown>, init?: RequestInit) => {
  const headers = new Headers(init?.headers)
  let payload: BodyInit | undefined

  if (body instanceof FormData) {
    payload = body
  } else if (body !== undefined) {
    headers.set('Content-Type', 'application/json')
    payload = JSON.stringify(body)
  }

  const token = getAuthToken()
  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`)
  }

  const response = await fetch(buildUrl(path), {
    ...init,
    method,
    headers,
    body: payload,
  })

  if (!response.ok) {
    const message = await parseErrorMessage(response)
    throw new Error(message || 'Request failed')
  }

  if (response.status === 204) {
    return undefined as T
  }

  const contentType = response.headers.get('content-type')
  if (contentType && contentType.includes('application/json')) {
    return (await response.json()) as T
  }

  return (await response.text()) as T
}

export const api = {
  get: <T>(path: string, init?: RequestInit) => request<T>('GET', path, undefined, init),
  post: <T>(path: string, body?: BodyInit | Record<string, unknown>, init?: RequestInit) =>
    request<T>('POST', path, body, init),
  put: <T>(path: string, body?: BodyInit | Record<string, unknown>, init?: RequestInit) =>
    request<T>('PUT', path, body, init),
  delete: <T>(path: string, init?: RequestInit) => request<T>('DELETE', path, undefined, init),
}

export type ApiClient = typeof api
