FROM ubuntu:16.04
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
tensorflow==1.8.0 mxnet==1.2.0 keras==2.2.0 keras-mxnet==2.2.0

RUN ln -s /usr/bin/python3.6 /usr/bin/python
