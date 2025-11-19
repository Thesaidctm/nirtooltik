import * as XLSX from 'xlsx'

const sanitizeNumericRow = (row: unknown[]): number[] =>
  row
    .map((value) => {
      if (typeof value === 'number') return value
      if (typeof value === 'string') {
        const parsed = Number(value.trim())
        return Number.isFinite(parsed) ? parsed : null
      }
      return null
    })
    .filter((value): value is number => value !== null)

const parseDelimitedText = (text: string): number[][] => {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)

  return lines
    .map((line) => line.split(/[;,\t]+/).map((value) => Number(value.trim())).filter((value) => Number.isFinite(value)))
    .filter((row) => row.length > 0)
}

export const parseSpectraFromText = (text: string): number[][] => {
  return parseDelimitedText(text)
}

export const parseSpectraFile = async (file: File): Promise<number[][]> => {
  const extension = file.name.toLowerCase().split('.').pop()

  const loadAsText = (): Promise<string> =>
    new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => {
        if (typeof reader.result === 'string') {
          resolve(reader.result)
        } else {
          reject(new Error('Não foi possível ler o arquivo selecionado'))
        }
      }
      reader.onerror = () => reject(new Error('Falha ao processar o arquivo'))
      reader.readAsText(file)
    })

  const loadAsArrayBuffer = (): Promise<ArrayBuffer> =>
    new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => {
        if (reader.result instanceof ArrayBuffer) {
          resolve(reader.result)
        } else {
          reject(new Error('Não foi possível ler o arquivo selecionado'))
        }
      }
      reader.onerror = () => reject(new Error('Falha ao processar o arquivo'))
      reader.readAsArrayBuffer(file)
    })

  if (extension === 'csv' || extension === 'txt') {
    const text = await loadAsText()
    return parseDelimitedText(text)
  }

  if (extension === 'xlsx' || extension === 'xls') {
    const buffer = await loadAsArrayBuffer()
    const workbook = XLSX.read(buffer, { type: 'array' })
    const sheetName = workbook.SheetNames[0]
    const worksheet = workbook.Sheets[sheetName]
    const rows = XLSX.utils.sheet_to_json(worksheet, { header: 1, raw: true }) as unknown[][]

    return rows
      .map((row) => sanitizeNumericRow(row))
      .filter((row) => row.length > 0)
  }

  throw new Error('Formato de arquivo não suportado. Use CSV ou XLSX.')
}
