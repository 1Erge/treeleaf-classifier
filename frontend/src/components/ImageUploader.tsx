
import { useRef, useState } from 'react'
import {
  Upload,
  Image as ImageIcon,
  X,
  Plus,
  Leaf,
} from 'lucide-react'

interface ImageUploaderProps {
  onFilesChange: (files: File[]) => void
}

const MAX_FILES = 20
const MAX_FILE_SIZE = 10 * 1024 * 1024

const SUPPORTED_TYPES = [
  'image/jpeg',
  'image/png',
  'image/webp',
  'image/bmp',
  'image/tiff',
]

function ImageUploader({ onFilesChange }: ImageUploaderProps) {
  const [files, setFiles] = useState<File[]>([])
  const [dragActive, setDragActive] = useState(false)
  const [error, setError] = useState('')

  const inputRef = useRef<HTMLInputElement>(null)

  const updateFiles = (newFiles: File[]) => {
    setError('')

    const validFiles: File[] = []
    const errors: string[] = []

    for (const file of newFiles) {
      if (!SUPPORTED_TYPES.includes(file.type)) {
        errors.push(`${file.name}: неподдерживаемый формат`)
        continue
      }

      if (file.size > MAX_FILE_SIZE) {
        errors.push(`${file.name}: размер превышает 10 МБ`)
        continue
      }

      validFiles.push(file)
    }

    const uniqueFiles = validFiles.filter(
      (file) =>
        !files.some(
          (existing) =>
            existing.name === file.name &&
            existing.size === file.size &&
            existing.lastModified === file.lastModified,
        ),
    )

    const availableSlots = MAX_FILES - files.length
    const selectedFiles = uniqueFiles.slice(0, availableSlots)

    if (uniqueFiles.length > availableSlots) {
      errors.push('Можно загрузить не более 20 изображений')
    }

    const updatedFiles = [...files, ...selectedFiles]

    setFiles(updatedFiles)
    onFilesChange(updatedFiles)

    if (errors.length > 0) {
      setError(errors.join('. '))
    }
  }

  const removeFile = (index: number) => {
    const updatedFiles = files.filter((_, i) => i !== index)

    setFiles(updatedFiles)
    onFilesChange(updatedFiles)
    setError('')
  }

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragActive(false)

    updateFiles(Array.from(event.dataTransfer.files))
  }

  const handleInputChange = (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    if (event.target.files) {
      updateFiles(Array.from(event.target.files))
    }

    event.target.value = ''
  }

  return (
    <div className="w-full">
      <div
        onDragOver={(event) => {
          event.preventDefault()
          setDragActive(true)
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`cursor-pointer rounded-3xl border-2 border-dashed p-10 text-center transition ${
          dragActive
            ? 'border-green-600 bg-green-100'
            : 'border-green-200 bg-green-50/50 hover:border-green-500 hover:bg-green-50'
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/bmp,image/tiff"
          multiple
          className="hidden"
          onChange={handleInputChange}
        />

        <div className="mx-auto mb-5 flex h-20 w-20 items-center justify-center rounded-2xl bg-green-100 text-green-700">
          <Upload size={36} />
        </div>

        <h3 className="text-xl font-semibold text-gray-900">
          Загрузите изображения листьев
        </h3>

        <p className="mt-3 text-sm text-gray-500">
          Перетащите изображения сюда или нажмите для выбора файлов
        </p>

        <p className="mt-2 text-xs text-gray-400">
          JPEG, PNG, WEBP, BMP, TIFF · До 10 МБ на изображение
        </p>

        <div className="mt-6">
          <span className="inline-flex items-center gap-2 rounded-xl bg-green-700 px-6 py-3 font-medium text-white transition hover:bg-green-800">
            <ImageIcon size={18} />
            Выбрать изображения
          </span>
        </div>
      </div>

      {error && (
        <div className="mt-4 rounded-xl bg-red-50 p-4 text-sm text-red-700">
          {error}
        </div>
      )}

      {files.length > 0 && (
        <div className="mt-8">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="font-semibold text-gray-900">
              Выбранные изображения
            </h3>

            <span className="text-sm text-gray-500">
              {files.length} из {MAX_FILES}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4">
            {files.map((file, index) => (
              <div
                key={`${file.name}-${file.lastModified}`}
                className="group relative overflow-hidden rounded-2xl border border-gray-200 bg-white"
              >
                <img
                  src={URL.createObjectURL(file)}
                  alt={file.name}
                  className="h-36 w-full object-cover"
                />

                <button
                  type="button"
                  onClick={(event) => {
                    event.stopPropagation()
                    removeFile(index)
                  }}
                  className="absolute right-2 top-2 rounded-full bg-white p-2 text-gray-600 shadow transition hover:bg-red-50 hover:text-red-600"
                  aria-label="Удалить изображение"
                >
                  <X size={16} />
                </button>

                <p className="truncate px-3 py-2 text-xs text-gray-600">
                  {file.name}
                </p>
              </div>
            ))}

            {files.length < MAX_FILES && (
              <button
                type="button"
                onClick={() => inputRef.current?.click()}
                className="flex min-h-44 flex-col items-center justify-center rounded-2xl border-2 border-dashed border-gray-200 text-gray-400 transition hover:border-green-400 hover:text-green-700"
              >
                <Plus size={30} />
                <span className="mt-2 text-sm">
                  Добавить ещё
                </span>
              </button>
            )}
          </div>
        </div>
      )}

      {files.length === 0 && (
        <div className="mt-6 flex items-center justify-center gap-2 text-sm text-gray-400">
          <Leaf size={16} />
          <span>Для начала работы загрузите фотографию листа</span>
        </div>
      )}
    </div>
  )
}

export default ImageUploader
