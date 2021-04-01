FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel

# mujoco
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    wget \
    xpra \
    xserver-xorg-dev
RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf
RUN mkdir -p /root/.mujoco && \
    wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip && \
    unzip mujoco.zip -d /root/.mujoco && \
    rm mujoco.zip
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin' >> /root/.bashrc

# install python libraries
COPY setups/ /root/setups/