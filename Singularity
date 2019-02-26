Bootstrap: docker
From: ubuntu

%help
Singularity container for autonet

%labels
    Version v0.1

%environment
    export PATH=/data/miniconda/bin:$PATH

%setup
    mkdir ${SINGULARITY_ROOTFS}/data

%post
    apt-get update
    apt-get -y install wget git gcc g++ tar libpython3.6-dev
    cd /data
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /data/miniconda
    export PATH=/data/miniconda/bin:$PATH

    conda install --yes python=3.6
    conda install --yes -c anaconda gcc_linux-64 gxx_linux-64
    conda install --yes pip wheel nose swig mkl-service

    git clone https://github.com/urbanmatthias/auto-sklearn.git
    cd auto-sklearn
    git checkout autonet

    pip install numpy==1.12.0
    pip install smac==0.8.0
    cat requirements.txt | xargs -n 1 -L 1 pip install
    pip install xmltodict requests liac-arff openml xgboost==0.80
    pip install theano==0.9.0
    pip install git+https://github.com/Lasagne/Lasagne@5aadc1b0fa61ad2d27aabbfc6392c5d388e5bba5
