FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
# 9.0-cudnn7-runtime-ubuntu16.04

ENV DEBIAN_FRONTEND noninteractive

# install essentials
RUN apt-get update -y && apt-get install -y \
	software-properties-common \
	build-essential \
	libblas-dev \
	libhdf5-serial-dev \
	python3-dev \
	python3-pip \
	git

# install some nice tools to help with development inside the image
RUN apt-get update -y && apt-get install -y zsh tmux htop vim

# of course, pip, plus and making python3 the default
RUN pip3 install -U pip
RUN ln -s /usr/bin/python3 /usr/bin/python

# make terminals look pretty (setting a reasonable colour setting)
RUN touch /usr/share/locale/locale.alias
RUN apt-get -y install locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8
ENV LANG en_US.UTF-8
ENV TERM xterm-256color

# change workdir and add some files
WORKDIR /src
ADD ./requirements.txt ./
ADD ./git-requirements.txt ./

# install our beloved requirements
RUN pip install -r requirements.txt
RUN pip install -r git-requirements.txt
RUN apt-get autoremove -y && apt-get autoclean -y