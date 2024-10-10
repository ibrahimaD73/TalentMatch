
FROM python:3.9 AS builder

WORKDIR /app


RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


FROM python:3.9-slim

WORKDIR /app


COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin


COPY . .
s
RUN mkdir -p uploads/candidatures uploads/offres

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]