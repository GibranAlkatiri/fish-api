FROM python:3.10-slim

# Install libGL untuk OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Install pip dependencies
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8080

# Jalankan aplikasi Flask
CMD ["python", "app.py"]