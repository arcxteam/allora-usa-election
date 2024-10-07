# Use an official Python runtime
FROM python:3.11-slim AS project_env

# Install curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools \
    && pip install --no-cache-dir -r requirements.txt

FROM project_env

COPY . /app/

# Set the entrypoint command
CMD ["gunicorn", "--conf", "/app/gunicorn_conf.py", "main:app"]
