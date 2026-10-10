# Python environment of the method (CPU): Python 3.10.16, PyTorch 1.11.0, NumPy 1.24.4, pandas 2.3.3,
# scikit-learn 1.7.0, Scanpy 1.11.3, anndata 0.11.4, SciPy 1.15.3, POT 0.9.5.
FROM python:3.10.16-slim-bookworm

ARG UID=1000
ARG GID=1000
ARG PANDAS_VERSION=2.3.3
ARG SCIPY_VERSION=1.15.3

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git tini procps \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install torch==1.11.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu \
 && pip install -r /tmp/requirements.txt "pandas==${PANDAS_VERSION}" "scipy==${SCIPY_VERSION}" \
 && pip check

RUN groupadd -g ${GID} dev \
 && useradd -m -u ${UID} -g ${GID} -s /bin/bash dev \
 && mkdir -p /workspace && chown dev:dev /workspace
USER dev
ENV HOME=/home/dev \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    MPLCONFIGDIR=/tmp/mplconfig \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/src
WORKDIR /workspace
ENTRYPOINT ["tini", "--"]
CMD ["bash"]
