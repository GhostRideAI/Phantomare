FROM ubuntu:22.04
USER root
RUN apt update && apt upgrade -y

# Setup Docker User
RUN apt update && apt install sudo
RUN adduser --gecos "" --disabled-password docker-user && adduser docker-user sudo
COPY entrypoint.sh /root/
ENTRYPOINT /root/entrypoint.sh
RUN echo 'docker-user:Docker!' | chpasswd

# Install miniconda
RUN apt update \
    && apt install -y build-essential \
    && apt install -y wget \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*
ENV CONDA_DIR /home/docker-user/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /home/docker-user/miniconda.sh
USER docker-user
RUN /bin/bash ~/miniconda.sh -b -p ~/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda init bash

# Install misc. tools
USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install ffmpeg libsm6 libxext6 xvfb -y
RUN apt install git -y

# Create Conda env. & install dependencies
USER docker-user
RUN conda create --name Phantomare python=3.11
COPY requirements.txt /home/docker-user/
# Work around for box2d-py error
RUN conda run -n Phantomare pip install swig==4.2.1
RUN conda run -n Phantomare pip install -r ~/requirements.txt
RUN echo 'conda activate Phantomare' >> ~/.bashrc

# Defined in requirements.txt
#RUN conda run -n Phantomare pip install -U ray[default] ray[tune]
#RUN conda run -n Phantomare pip install -U imageio imageio-ffmpeg
#RUN conda run -n Phantomare pip install -U numpy
#RUN conda run -n Phantomare pip install -U pandas
#RUN conda run -n Phantomare pip install -U pillow
#RUN conda run -n Phantomare pip install -U torch torchvision pytorch-lightning
#RUN conda run -n Phantomare pip install -U moviepy
#RUN conda run -n Phantomare pip install -U matplotlib
#RUN conda run -n Phantomare pip install -U tensorboard tensorboardx
#RUN conda run -n Phantomare pip install -U scikit-image
#RUN conda run -n Phantomare pip install -U ipykernel
#RUN conda run -n Phantomare pip install -U opencv-python
#RUN conda run -n Phantomare pip install -U torch-pruning
#RUN conda run -n Phantomare pip install -U tensordict
#RUN conda run -n Phantomare pip install -U torchinfo
#RUN conda run -n Phantomare pip install -U torch-tb-profiler
#RUN conda run -n Phantomare pip install -U torchrl
#RUN conda run -n Phantomare pip install -U swig
#RUN conda run -n Phantomare pip install -U gymnasium[box2d]
#RUN conda run -n Phantomare pip install -U pytest
#RUN conda run -n Phantomare pip install -U shapely
#RUN conda run -n Phantomare pip install -U scikit-learn

# Update PS1
RUN echo 'export PS1="\[$(tput bold)\]\[\033[38;5;105m\]Phantomare \[☁️\]\[$(tput sgr0)\]  > \[$(tput sgr0)\]"' >> ~/.bashrc
