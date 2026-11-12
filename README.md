# TreeLeaf Classifier — этап 1

Подготовка проекта, загрузка Flavia и построение внутреннего представления данных.

## Требования

- Python 3.10+
- Свободное место: не менее 2–3 ГБ (архив Flavia около 966 МБ, плюс распакованные изображения)
- Интернет для загрузки датасета

## 1. Создать окружение

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell
```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Скачать Flavia

Из корня проекта:
```bash
python scripts/download_flavia.py
```

Скрипт сохраняет архив в `data/downloads/Leaves.tar.bz2` и проверяет SHA-256.

Если архив уже скачан вручную:
```bash
python scripts/download_flavia.py --skip-download
```

Официальный источник:
https://sourceforge.net/projects/flavia/files/Leaf%20Image%20Dataset/1.0/

## 3. Распаковать датасет

```bash
python scripts/extract_flavia.py
```

По умолчанию изображения будут расположены в `data/raw/Leaves/`.
Скрипт безопасно проверяет пути внутри архива, не извлекает ссылки и специальные файлы.

Если архив содержит дополнительную вложенную директорию, сканер всё равно
найдёт JPG-файлы рекурсивно.

## 4. Построить манифест и статистику

```bash
python scripts/scan_dataset.py
```

Результаты:
- `data/manifests/flavia_manifest.csv` — одна строка на изображение;
- `data/manifests/flavia_summary.json` — статистика классов и ошибок.

Можно указать другую директорию:
```bash
python scripts/scan_dataset.py --root "D:/datasets/Leaves"
```

## 5. Проверить результат

Ожидается примерно 1 907 изображений и 32 класса. Фактическое количество
скрипт покажет в summary. Не все версии/зеркала датасета обязательно
содержат идентичный набор файлов.

## Структура

```text
treeleaf-classifier/
├── data/
│   ├── downloads/
│   ├── raw/
│   └── manifests/
├── scripts/
│   ├── download_flavia.py
│   ├── extract_flavia.py
│   └── scan_dataset.py
├── src/
│   └── treeleaf/
│       └── data/
│           ├── classes.py
│           ├── loader.py
│           └── manifest.py
├── requirements.txt
└── README.md
```

## Важное замечание о метках Flavia

В оригинальном Flavia имена изображений — четырёхзначные числовые ID.
Класс определяется диапазоном ID из официальной таблицы датасета, а не
путём угадывания по имени папки. В исходной таблице номера классов идут
с пропуском 13: фактически представлено 32 класса. В проекте каждому
классу дополнительно назначается непрерывный `class_index` от 0 до 31,
который будет использоваться нейросетью.

При публикации/использовании датасета указывайте исходную работу:
Stephen Gang Wu et al., “A Leaf Recognition Algorithm for Plant
classification Using Probabilistic Neural Network”, IEEE 7th International
Symposium on Signal Processing and Information Technology, 2007.
