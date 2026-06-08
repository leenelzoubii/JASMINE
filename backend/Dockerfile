# Use the full Debian-based image instead of slim
FROM python:3.11-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libegl1 \
    libgles2 \
    libglib2.0-0 \
    libgomp1 \
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

# Force the system to find the graphics libraries
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]