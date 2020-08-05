FROM nvidia/cuda:9.1-base
LABEL author=DanielJunior email="danieljunior@id.uff.br"
USER root

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates \
    gcc g++ nano cython build-essential \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git 

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "source activate base" > ~/.bashrc

RUN mkdir -p /app
WORKDIR /app
RUN pip install --upgrade pip

EXPOSE 8888