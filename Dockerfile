FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    python=python3.8

RUN mkdir -p /usr/share/man/man1 && \
    apt-get update && apt-get install -y \
    build-essential \
    cifs-utils \
    curl \
    default-jdk \
    dialog \
    dos2unix \
    git \
    sudo \
    wget \
    unzip \
    nano \
    pip \
    python3.8


# Install app requirements first to avoid invalidating the cache
WORKDIR /app

# Copy all files
COPY requirements.txt setup.py /app/

# Install packages
RUN pip install torch==1.5.0 torchtext==0.3.1 torchvision==0.11.1 torchaudio==0.10.0 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install -r requirements.txt && \
    ${python} -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Assume that the datasets will be mounted as a volume into /mnt/data on startup.
# Symlink the data subdirectory to that volume.
ENV CACHE_DIR=/mnt/data
RUN mkdir -p /mnt/data && \
    ln -snf /mnt/data /app/data

# Copy all the rest
COPY . .

# Convert all shell scripts to Unix line endings, if any
RUN /bin/bash -c 'if compgen -G "/app/**/*.sh" > /dev/null; then dos2unix /app/**/*.sh; fi'

ENTRYPOINT bash
