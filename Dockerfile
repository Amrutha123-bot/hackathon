# Use a base Python image. Choose a version that matches your development environment
# and is compatible with your dependencies (e.g., 3.10, 3.11).
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed by libraries like PyMuPDFLoader and unstructured
# poppler-utils is crucial for PDF processing.
# build-essential is often needed for compiling Python packages with C extensions.
# python3-pip and python3-venv ensure pip and venv are available.
# libmagic1 is used by unstructured for file type identification.
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    python3-pip \
    python3-venv \
    build-essential \
    libmagic1 \
    # Add other system dependencies here if needed for other file types or libraries
    # e.g., for Tesseract OCR if unstructured uses it: tesseract-ocr
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
# Ensure your requirements.txt is correctly formatted and includes all necessary Python packages.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
# This includes your main.py, any utility scripts, and potentially the chroma_db folder if you were pre-building it (but we are not doing that here for ephemeral deployment).
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit application
# Ensure 'main.py' is the correct entry point for your Streamlit app.
CMD ["streamlit", "run", "main.py"]