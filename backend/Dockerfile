FROM python:3.11-bookworm

WORKDIR /app

# 1. Install system dependencies (Slow but rarely changes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libegl1 libgles2 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy ONLY the requirements first
COPY requirements.txt .

# 3. Install requirements (Only runs if requirements.txt changes)
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the code (Runs every time, but very fast)
COPY . .

# Fix links and envs
RUN ln -s /usr/lib/x86_64-linux-gnu/libGLESv2.so.2 /usr/lib/x86_64-linux-gnu/libGLESv2.so || true
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]