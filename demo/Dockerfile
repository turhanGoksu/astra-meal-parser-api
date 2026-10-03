# Hugging Face Docker Space: the same image runs locally and on the Space.
#   docker build -t astra-nutrition-demo demo/
#   docker run --rm -p 127.0.0.1:7860:7860 astra-nutrition-demo
FROM python:3.12-slim

# llama-cpp-python compiles llama.cpp during install (libgomp at runtime).
# GGML_NATIVE=OFF: the Space may build the image on one CPU and run it on
# another, and a build tuned for the first can crash on the second.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake libgomp1 \
    && rm -rf /var/lib/apt/lists/*
ENV CMAKE_ARGS=-DGGML_NATIVE=OFF \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# Spaces run the container as uid 1000; the model goes to its HF cache.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt
COPY --chown=user app.py .

EXPOSE 7860
CMD ["python", "app.py"]
