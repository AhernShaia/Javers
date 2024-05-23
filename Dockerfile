# 使用 python:3.12-slim 映像作為基礎映像
FROM python:3.12-slim as base

# 創建一個名為 builder 的新階段，並使用 base 映像作為基礎映像
FROM base as builder

# 安裝 Poetry，這是一個 Python 的套件管理工具
RUN pip install poetry==1.4.2

# 設定 Poetry 的環境變數
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# 設定工作目錄為 /app
WORKDIR /app

# 複製 pyproject.toml 和 poetry.lock 到工作目錄
COPY pyproject.toml poetry.lock ./

# 創建一個名為 README.md 的空檔案
RUN touch README.md

# 使用 Poetry 安裝 Python 套件，並將 Poetry 的快取目錄掛載為 Docker 的快取
RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --without dev --no-root

# 創建一個新的階段，並使用 base 映像作為基礎映像
FROM base

# 設定 Python 虛擬環境的路徑，並將其添加到 PATH 環境變數
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# 從 builder 階段複製 Python 虛擬環境到當前映像
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# 複製 bot 目錄下的所有檔案到當前映像的 /app 目錄
COPY ./bot/ ./app

# 設定工作目錄為 /app
WORKDIR /app

# 當 Docker 容器啟動時，執行 Uvicorn 伺服器
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
