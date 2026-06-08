FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV + MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy EVERYTHING in the current folder (main.py, src/, models/, etc.) into /app
COPY . /app

# Install backend Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly tell Python where the root is so src imports work perfectly
ENV PYTHONPATH="/app"

# Hugging Face Spaces expects port 7860
ENV PORT=7860
EXPOSE 7860

# Start the FastAPI server directly from the root main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]