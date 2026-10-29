# C2 Detection API — Docker image
# Build:  docker build -t iomt-c2 -f c2_detection/Dockerfile .
# Run:    docker run -p 8002:8002 iomt-c2
# With API key:  docker run -p 8002:8002 -e C2_API_KEY=secret iomt-c2
 
FROM python:3.11-slim
 
WORKDIR /app
 
# Install dependencies first (cached layer)
COPY c2_detection/requirements.txt c2_detection/requirements.txt
RUN pip install --no-cache-dir -r c2_detection/requirements.txt
 
# Copy source
COPY contracts/ contracts/
COPY c2_detection/ c2_detection/
 
# Expose API port
EXPOSE 8002
 
# Start FastAPI
CMD ["uvicorn", "c2_detection.api.main:app", "--host", "0.0.0.0", "--port", "8002"]
 