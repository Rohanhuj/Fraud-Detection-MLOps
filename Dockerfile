FROM python:3.11-slim

WORKDIR /app

COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY src ./src
COPY artifacts ./artifacts

EXPOSE 8000

CMD ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
