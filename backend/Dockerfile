FROM python:3.12-slim

ARG TARGETARCH

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deno.land/install.sh | sh
ENV DENO_INSTALL="/root/.deno"
ENV PATH="$DENO_INSTALL/bin:$PATH"

COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir cython "numpy<2.0" \
    && pip install --no-cache-dir --no-build-isolation -r requirements.txt

# TensorFlow wheels are not published for linux/arm64; only install on amd64.
RUN if [ "$TARGETARCH" = "amd64" ]; then \
      pip install --no-cache-dir --no-build-isolation -r requirements-ml.txt; \
    else \
      echo "Skipping basic-pitch/tensorflow on $TARGETARCH"; \
    fi

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
