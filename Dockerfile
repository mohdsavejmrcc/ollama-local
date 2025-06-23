FROM python:3.12

# Set working directory
WORKDIR /app

# Install system dependencies (for packages like `cffi`, `psycopg2`, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libffi-dev \
    python3-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first (to use Docker cache)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run Django server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
