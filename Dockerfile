FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY web/ ./web/
COPY docker-entrypoint.sh ./

EXPOSE 8000

ENTRYPOINT ["sh", "/app/docker-entrypoint.sh"]
