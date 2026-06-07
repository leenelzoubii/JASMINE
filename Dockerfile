FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV + MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend code
COPY jasmine-next/backend/ /app/jasmine-next/backend/

# Copy ML pipeline (imported by backend via sys.path)
COPY src/ /app/src/

# Copy trained model files
COPY models/ /app/models/

# Install backend Python dependencies
RUN pip install --no-cache-dir -r /app/jasmine-next/backend/requirements.txt

# Hugging Face Spaces expects port 7860
ENV PORT=7860
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "jasmine-next.backend.main:app", "--host", "0.0.0.0", "--port", "7860"]