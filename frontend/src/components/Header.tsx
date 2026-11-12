
import { Leaf } from 'lucide-react'

function Header() {
  return (
    <header className="border-b border-green-100 bg-white">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
        <a href="/" className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-green-700 text-white">
            <Leaf size={25} />
          </div>

          <div>
            <h1 className="text-xl font-bold text-green-900">
              TreeLeaf
            </h1>
            <p className="text-xs text-gray-500">
              Определение видов деревьев
            </p>
          </div>
        </a>

        <div className="hidden items-center gap-2 rounded-full bg-green-50 px-4 py-2 text-sm text-green-800 sm:flex">
          <span className="h-2 w-2 rounded-full bg-green-500" />
          Искусственный интеллект
        </div>
      </div>
    </header>
  )
}

export default Header
