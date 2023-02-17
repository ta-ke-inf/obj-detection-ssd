FROM nvidia/cuda:11.7.0-base-ubuntu22.04
RUN apt-get update && apt-get install -y \
    python3-pip \
    sudo \
    wget \
    git \
    vim \
    libgl1-mesa-dev \
    libgtk2.0-dev \
    libsndfile1

RUN ln -s /usr/bin/python3.10 /usr/bin/python
RUN pip install --upgrade pip
RUN pip install \
    torch \
    torchvision \
    torchaudio \
    jupyterlab \
    matplotlib \
    tqdm \
    opencv-python \
    librosa \
    optuna \
    pandas \
    seaborn

WORKDIR /work
#CMD ["/bin/bash"]
#CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--LabApp.token=''"]
