FROM node:22-alpine AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.13-slim

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.14.1 torchvision==0.29.1
COPY requirements-inference.txt ./
RUN pip install --no-cache-dir -r requirements-inference.txt
RUN python -c "from torchvision.models import ResNet18_Weights, resnet18; resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)"
COPY requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements-web.txt
COPY stratscout/ ./stratscout/
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "stratscout.webapp:app", "--host", "0.0.0.0", "--port", "8000"]
