# Use Python 3.12 as the base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt from the root directory to /app/
COPY requirements.txt /app/

# Install dependencies from requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Copy the rest of the application files into the container
COPY . /app/

# Expose the Flask port 
EXPOSE 5000

# Run the Flask application
CMD ["python", "app.py"]
