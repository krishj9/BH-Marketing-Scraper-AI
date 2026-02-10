FROM python:3.11-slim

# Install system dependencies for Playwright Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libxshmfence1 \
    libgtk-3-0 \
    libx11-xcb1 \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright and Chromium browser
RUN playwright install chromium && playwright install-deps chromium

# Copy application code and config
COPY src/ src/
COPY config/ config/

# Default environment variables (override in Cloud Run)
ENV RUN_MODE=cloud_run
ENV GCP_PROJECT_ID=mybrightday-dev
ENV GCP_REGION=us-east4
ENV BIGQUERY_DATASET=bh_scraper

CMD ["python", "-m", "src.main"]
