# Use an official lightweight Python image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required for certain python libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements or individual files
COPY app.py /app/
COPY docs/ /app/docs/

# We'll install the python packages directly to keep it simple
RUN pip install --no-cache-dir \
    streamlit \
    langchain \
    langchain-community \
    langchain-chroma \
    pypdf \
    ollama

# Expose the default Streamlit port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
