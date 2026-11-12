
import { useState } from 'react'
import {
  Leaf,
  Sparkles,
  ShieldCheck,
  LoaderCircle,
  AlertCircle,
} from 'lucide-react'

import Header from './components/Header'
import ImageUploader from './components/ImageUploader'
import PredictionResults from './components/PredictionResults'

import {
  predictImages,
  type ImagePrediction,
} from './api/predictions'

function App() {
  const [files, setFiles] = useState<File[]>([])
  const [results, setResults] = useState<ImagePrediction[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleFilesChange = (newFiles: File[]) => {
    setFiles(newFiles)
    setResults([])
    setError('')
  }

  const handlePrediction = async () => {
    if (files.length === 0 || loading) {
      return
    }

    setLoading(true)
    setError('')
    setResults([])

    try {
      const response = await predictImages(files)

      setResults(response.predictions)
    } catch (err) {
      console.error('Ошибка классификации:', err)

      if (typeof err === 'object' && err !== null && 'response' in err) {
        const response = (
          err as {
            response?: {
              data?: {
                detail?: string
              }
            }
          }
        ).response

        setError(
          response?.data?.detail ||
          'Не удалось обработать изображения.',
        )
      } else {
        setError(
          'Не удалось подключиться к серверу. Проверьте, запущен ли FastAPI.',
        )
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-[#f7faf7]">
      <Header />

      <main className="mx-auto max-w-7xl px-6 py-12">
        <section className="mx-auto max-w-3xl text-center">
          <div className="mb-5 inline-flex items-center gap-2 rounded-full bg-green-100 px-4 py-2 text-sm font-medium text-green-800">
            <Sparkles size={16} />
            Интеллектуальное распознавание растений
          </div>

          <h1 className="text-4xl font-bold tracking-tight text-gray-900 sm:text-5xl">
            Узнайте, какое дерево
            <span className="block text-green-700">
              перед вами
            </span>
          </h1>

          <p className="mx-auto mt-6 max-w-2xl text-lg leading-8 text-gray-500">
            Загрузите фотографию листа, и наша модель искусственного
            интеллекта определит его вид среди 32 видов деревьев.
          </p>
        </section>

        <section className="mx-auto mt-12 max-w-4xl rounded-3xl border border-green-100 bg-white p-6 shadow-sm sm:p-10">
          <div className="mb-8">
            <h2 className="text-2xl font-bold text-gray-900">
              Определение вида дерева
            </h2>

            <p className="mt-2 text-sm text-gray-500">
              Загрузите одно или несколько изображений листьев
            </p>
          </div>

          <ImageUploader onFilesChange={handleFilesChange} />

          {error && (
            <div className="mt-6 flex items-start gap-3 rounded-xl border border-red-100 bg-red-50 p-4 text-sm text-red-700">
              <AlertCircle className="mt-0.5 shrink-0" size={20} />
              <p>{error}</p>
            </div>
          )}

          <div className="mt-8 flex flex-col items-center">
            <button
              type="button"
              onClick={handlePrediction}
              disabled={files.length === 0 || loading}
              className="flex w-full max-w-sm items-center justify-center gap-3 rounded-xl bg-green-700 px-8 py-4 font-semibold text-white transition hover:bg-green-800 disabled:cursor-not-allowed disabled:bg-gray-300"
            >
              {loading ? (
                <>
                  <LoaderCircle className="animate-spin" size={20} />
                  Анализируем изображения...
                </>
              ) : (
                <>
                  <Sparkles size={20} />
                  Определить вид дерева
                </>
              )}
            </button>

            <p className="mt-3 text-xs text-gray-400">
              Выбрано изображений: {files.length}
            </p>
          </div>
        </section>

        <PredictionResults results={results} />

        <section className="mx-auto mt-12 grid max-w-4xl gap-5 sm:grid-cols-3">
          <div className="flex items-center gap-3 rounded-2xl bg-white p-5 shadow-sm">
            <div className="rounded-xl bg-green-100 p-3 text-green-700">
              <Leaf size={22} />
            </div>
            <div>
              <p className="font-semibold text-gray-900">32 вида</p>
              <p className="text-xs text-gray-500">Деревьев в модели</p>
            </div>
          </div>

          <div className="flex items-center gap-3 rounded-2xl bg-white p-5 shadow-sm">
            <div className="rounded-xl bg-green-100 p-3 text-green-700">
              <Sparkles size={22} />
            </div>
            <div>
              <p className="font-semibold text-gray-900">ResNet50</p>
              <p className="text-xs text-gray-500">Архитектура модели</p>
            </div>
          </div>

          <div className="flex items-center gap-3 rounded-2xl bg-white p-5 shadow-sm">
            <div className="rounded-xl bg-green-100 p-3 text-green-700">
              <ShieldCheck size={22} />
            </div>
            <div>
              <p className="font-semibold text-gray-900">99,65%</p>
              <p className="text-xs text-gray-500">Точность на тесте Flavia</p>
            </div>
          </div>
        </section>
      </main>

      <footer className="mt-16 border-t border-green-100 bg-white">
        <div className="mx-auto max-w-7xl px-6 py-6 text-center text-sm text-gray-400">
          TreeLeaf Classifier © 2026 · Определение видов деревьев
        </div>
      </footer>
    </div>
  )
}

export default App
