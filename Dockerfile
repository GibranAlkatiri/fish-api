FROM python:3.10-slim

# Install system dependencies untuk OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies terlebih dahulu (untuk caching layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh aplikasi
COPY . .

# Expose port (opsional, Railway akan mengoverride)
EXPOSE 8080

# Jalankan Gunicorn dengan PORT dari environment variable
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--workers", "2", "app:app"]