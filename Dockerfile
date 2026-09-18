FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-demo.txt ./
RUN pip install --no-cache-dir -r requirements-demo.txt
COPY . .
RUN git clone --depth 60 https://github.com/leon-ai/leon demo_repos/leon || true
ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 7860
CMD ["sh", "-c", "python scripts/fetch_cache.py; python app.py"]
