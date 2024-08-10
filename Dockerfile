FROM python:3.12

# DUST3R
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libglib2.0-0 \
    python3 \
    python3-dev \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --recursive https://github.com/tauzn-clock/dust3r /dust3r
WORKDIR /dust3r
RUN pip install -r requirements.txt
RUN pip install -r requirements_optional.txt
RUN pip install opencv-python==4.8.0.74