FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
LABEL maintainer="Hyungjun Kim <soeque1@gmail.com>"
ARG pyver=3.6

USER root

## setting Environments
RUN apt-get update && apt-get -yq dist-upgrade && \
apt-get install -yq --no-install-recommends \
wget git vim apt-transport-https \
bzip2 ssh cron

RUN mkdir /home/work

ENV SHELL=/bin/bash \
NB_USER=test_user \
NB_UID=1000 \
NB_GID=100 \
HOME=/home/$NB_USER

## python
RUN apt-get update && \
apt-get -y install --no-install-recommends software-properties-common && \
add-apt-repository ppa:jonathonf/python-${pyver} -y && \
add-apt-repository ppa:openjdk-r/ppa  && \
apt-get update && \
apt-get -y install --no-install-recommends python${pyver}-dev python${pyver}-venv python3-tk && \
apt-get -y install --no-install-recommends libopenblas-dev curl && \
apt-get -y install --no-install-recommends openjdk-8-jdk && \
apt-get clean && \
rm -rf /var/lib/apt/lists/* && \
curl https://bootstrap.pypa.io/get-pip.py | python${pyver}

## python libraries
RUN pip${pyver} install --upgrade pip && \
pip${pyver} install --no-cache-dir progressbar2 line_profiler \
tensorflow-gpu==1.8.0 mxnet-cu90mkl==1.2.0 keras==2.2.0 keras-mxnet==2.2.0

RUN ln -s /usr/bin/python3.6 /usr/bin/python
RUN wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.1.4.18-1+cuda9.0_amd64.deb -O /home/test_user/libcudnn7_7.1.4.18-1+cuda9.0_amd64.deb
RUN dpkg -i /home/test_user/libcudnn7_7.1.4.18-1+cuda9.0_amd64.deb

RUN ln -s /usr/bin/python3.6 /usr/bin/python
