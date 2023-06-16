FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
RUN apt update && \
    apt install -y \
    tzdata \
    curl \
    wget \
    sudo \
    vim \
    build-essential \
    libffi-dev \
    ca-certificates \
    libsqlite3-dev \
    apt-transport-https \
    python3-tk \
    git \
    libgl1-mesa-dev \
    zip \
    libxmlsec1-dev \
    liblzma-dev \
    tk-dev \
    llvm \
    libbz2-dev \
    libxml2-dev \
    systemd \
    lsof
    
# non-root USER settings
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser

RUN wget -P /tmp https://www.python.org/ftp/python/3.10.9/Python-3.10.9.tar.xz && cd /tmp && tar xJf Python-3.10.9.tar.xz
RUN cd /tmp/Python-3.10.9 && ./configure && make && sudo make install
RUN $(which python3) -m pip install -U pip setuptools wheel
RUN $(which python3) -m pip install \
    torch==1.13.1+cu116 \
    torchvision==0.14.1+cu116 \
    torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu116
RUN $(which python3) -m pip install \
    lightning pandas numpy scikit-learn \
    black flake8

WORKDIR /home/appuser/cable_bridge