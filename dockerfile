# Use an official Python 3.10 runtime as a parent image
FROM python:3.10-slim

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libc-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Set environment variables for Python
ENV PATH="/opt/venv/bin:$PATH"

# Create a virtual environment
RUN python -m venv /opt/venv

# Activate the virtual environment and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN . /opt/venv/bin/activate && pip install --upgrade pip && pip install wheel && pip install -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose port (if necessary, adjust to the port your application uses)
EXPOSE 8000

# Command to run the application
CMD ["/opt/venv/bin/python", "telebot.py"]
