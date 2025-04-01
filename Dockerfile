# Use the official Python image from Docker Hub
FROM python:3.12-alpine

RUN apk add --no-cache \
    gcc \
    musl-dev \
    libffi-dev \
    build-base \
    && pip install --upgrade pip
# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip and install required Python packages
RUN pip install --upgrade pip
RUN pip install  --no-cache-dir -r requirements.txt

# Copy the entire application (including app.py) into the working directory in the container
COPY . .

# Expose port 5000 to the outside world
EXPOSE 5000

# Set the command to run your app (this assumes your entry point is app.py)
CMD ["python", "app.py"]