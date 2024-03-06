FROM python:3.10-slim


RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends \
        # Required for downloading code/data from the internet \
        wget curl unzip git \
        # DevOps
        vim procps make \
        # Required by OpenCV
        libglib2.0-0 libgl1-mesa-glx \
        # Required for PostgreSQL \
        libpq-dev \
	build-essential \
   # Purge apt cache
   && apt-get clean \
   && rm -rf /var/lib/apt/lists/*

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade -r requirements.txt
RUN python -m pip install --no-cache-dir jupyterlab ipython

WORKDIR /app
COPY . /app

EXPOSE 8888
