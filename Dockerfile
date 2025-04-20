FROM python:3.10-slim

# Install dependencies yang dibutuhkan OpenCV (termasuk libGL dan libgthread)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Set working directory
WORKDIR /app

# Copy semua file ke dalam container
COPY . /app

# Install pip packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080
EXPOSE 8080

# Jalankan Flask
CMD ["python", "app.py"]