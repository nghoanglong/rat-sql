FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

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
    nano

# Install app requirements first to avoid invalidating the cache
WORKDIR /app

# Copy all files
COPY requirements.txt setup.py /app/

# Install packages
RUN pip install -r requirements.txt && \
    python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Cache the pretrained BERT model
RUN python -c "from transformers import BertModel; BertModel.from_pretrained('bert-large-uncased-whole-word-masking')"

# Download & cache embedding
RUN mkdir -p /app/third_party/phow2v_emb && \
    cd /app/third_party/phow2v_emb && \
    wget https://public.vinai.io/word2vec_vi_words_300dims.zip && \
    unzip word2vec_vi_words_300dims.zip

# Copy all the rest
COPY . .

# Convert all shell scripts to Unix line endings, if any
RUN /bin/bash -c 'if compgen -G "/app/**/*.sh" > /dev/null; then dos2unix /app/**/*.sh; fi'

ENTRYPOINT bash
