# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install AWS CLI
RUN apt-get update && apt-get install -y awscli

# Copy the rest of the application code into the container
COPY . /app

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the FastAPI app runs on
EXPOSE 8000

# Command to run the FastAPI application using uvicorn
CMD ["uvicorn", "adviser.app:app", "--host", "0.0.0.0", "--port", "8000"]