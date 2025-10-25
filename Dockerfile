#Use a Python base image

FROM python:3.11-slim-bookworm as builder

#environment variables
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

#installing required dependencies for MediaPipe and OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

#set working directory
WORKDIR /app

#copy and install dependencies; copying requirements.txt first
COPY requirements.txt .
#installing packages, ensuring MediaPipe and opencv-python are included
#--no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

#final runtime image
#smaller base image for final runtime to reduce the image size, inherit the installed packages

FROM python:3.11-slim-bookworm

#set working directory
WORKDIR /app

#install libGL.so.1 for OpenCV to work
RUN apt-get update && apt-get install -y libgl1-mesa-glx

#copy Python environment from the builder stage
#includes installed packages and compiled dependencies
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

COPY --from=builder /usr/local/bin /usr/local/bin

#copy application source code
COPY . .

#expose port for web service (Standardized to 5000 for Gunicorn/Render compatibility)
EXPOSE 10000

#CMD to run Flask application using Gunicorn (Binding to standard 5000 port)
CMD gunicorn --bind 0.0.0.0:${PORT:-10000} app:app
