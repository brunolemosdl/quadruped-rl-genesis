ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.3.2 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    QUADRUPED_RL_GENESIS_ARTIFACTS_ROOT=/outputs/artifacts \
    QUADRUPED_RL_GENESIS_OPTUNA_STORAGE=sqlite:////outputs/artifacts/optuna/studies.db \
    QUADRUPED_RL_GENESIS_DEVICE=cuda \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    curl \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libgles2 \
    libglvnd0 \
    libglib2.0-0 \
    libopengl0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==${POETRY_VERSION}"
RUN python -m venv "${VIRTUAL_ENV}" && pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY pyproject.toml poetry.lock README.md ./
COPY src ./src
COPY configs ./configs
COPY media ./media
COPY main.py ./
COPY .env.example ./

RUN poetry install --only main

VOLUME ["/outputs"]

ENTRYPOINT ["quadruped-rl-genesis", "benchmark"]
CMD ["--output-root", "/outputs", "--device", "cuda"]
