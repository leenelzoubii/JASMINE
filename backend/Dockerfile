FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libgles2 \
    && rm -rf /var/lib/apt/lists/*

# Copy ML pipeline source
COPY src/ /app/src/
COPY models/ /app/models/

# Copy backend
COPY main.py /app/main.py
COPY pose_extractor.py /app/pose_extractor.py
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# HF Spaces requires port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]