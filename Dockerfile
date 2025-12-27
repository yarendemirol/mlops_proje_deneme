# Python imajını çek
FROM python:3.9-slim

# Çalışma dizinini /app olarak ayarla
WORKDIR /app

# Alışveriş listeni (requirements) içeri kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tüm proje kodlarını ve model dosyanı içeri kopyala
COPY . .

# API'nin çalışacağı port
EXPOSE 8000

# Uygulamayı başlat (uvicorn üzerinden)
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]