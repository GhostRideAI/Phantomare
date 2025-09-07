# Start with Ubuntu 24.04 base
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
USER root
RUN apt update && apt upgrade -y

# Setup new user "Docker-User"
RUN apt update && apt install sudo adduser -y
RUN userdel -rf ubuntu
RUN adduser --gecos "" --disabled-password docker-user && adduser docker-user sudo
COPY entrypoint.sh /root/
ENTRYPOINT /root/entrypoint.sh
RUN echo 'docker-user:Docker!' | chpasswd

# Install system dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install git ffmpeg libsm6 libxext6 xvfb python3 python3-venv python3-dev gcc -y

# Create Python venv & install dependencies
RUN python3 -m venv /opt/venv
RUN chmod -R 777 /opt/venv
COPY requirements.txt /tmp/
RUN /opt/venv/bin/pip install -r /tmp/requirements.txt
ENV PATH="/opt/venv/bin:$PATH"

# Update PS1
RUN echo 'export PS1="👻 > "' >> /home/docker-user/.bashrc
