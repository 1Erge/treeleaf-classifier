import axios from 'axios'

const api = axios.create({
  baseURL: 'http://127.0.0.1:8000/api/v1',
  timeout: 120000,
})

export interface TopPrediction {
  class_name: string
  probability: number
}

export interface ImagePrediction {
  filename: string
  predicted_class: string
  confidence: number
  top_predictions: TopPrediction[]
}

export interface PredictionResponse {
  count: number
  predictions: ImagePrediction[]
}

export async function predictImages(
  files: File[],
): Promise<PredictionResponse> {
  const formData = new FormData()

  files.forEach((file) => {
    formData.append('files', file)
  })

  const response = await api.post<PredictionResponse>(
    '/predict',
    formData,
  )

  return response.data
}