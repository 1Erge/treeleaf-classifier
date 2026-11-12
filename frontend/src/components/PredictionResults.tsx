import { CheckCircle2, Leaf } from 'lucide-react'
import type { ImagePrediction } from '../api/predictions'
import { getTreeName } from '../utils/treeNames'

interface PredictionResultsProps {
  results: ImagePrediction[]
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(2)}%`
}

function PredictionResults({ results }: PredictionResultsProps) {
  if (results.length === 0) {
    return null
  }

  return (
    <section className="mx-auto mt-10 max-w-4xl">
      <div className="mb-6">
        <div className="flex items-center gap-2">
          <CheckCircle2 className="text-green-700" size={25} />

          <h2 className="text-2xl font-bold text-gray-900">
            Результаты определения
          </h2>
        </div>

        <p className="mt-2 text-sm text-gray-500">
          Обработано изображений: {results.length}
        </p>
      </div>

      <div className="space-y-5">
        {results.map((result, index) => (
          <article
            key={`${result.filename}-${index}`}
            className="rounded-2xl border border-green-100 bg-white p-6 shadow-sm"
          >
            <div className="flex flex-col gap-5 sm:flex-row sm:items-center">
              <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-green-100 text-green-700">
                <Leaf size={28} />
              </div>

              <div className="min-w-0 flex-1">
                <p className="truncate text-sm text-gray-500">
                  {result.filename}
                </p>

                <h3 className="mt-1 text-xl font-bold text-green-800">
                    {getTreeName(result.predicted_class)}
                </h3>

                <p className="mt-1 text-sm text-gray-500">
                  Уверенность модели: {formatPercent(result.confidence)}
                </p>
              </div>
            </div>

            <div className="mt-6 border-t border-gray-100 pt-5">
              <h4 className="mb-4 font-semibold text-gray-800">
                Топ-3 наиболее вероятных варианта
              </h4>

              <div className="space-y-4">
                {result.top_predictions.map((prediction, predictionIndex) => (
                  <div key={prediction.class_name}>
                    <div className="mb-2 flex items-center justify-between gap-3 text-sm">
                      <div className="flex items-center gap-2">
                        <span className="text-gray-400">
                          {predictionIndex + 1}.
                        </span>

                        <span className="font-medium text-gray-700">
                            {getTreeName(prediction.class_name)}
                        </span>
                      </div>

                      <span className="font-semibold text-green-800">
                        {formatPercent(prediction.probability)}
                      </span>
                    </div>

                    <div className="h-2 overflow-hidden rounded-full bg-gray-100">
                      <div
                        className="h-full rounded-full bg-green-600 transition-all duration-500"
                        style={{
                          width: `${Math.min(
                            Math.max(prediction.probability * 100, 0),
                            100,
                          )}%`,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </article>
        ))}
      </div>
    </section>
  )
}

export default PredictionResults