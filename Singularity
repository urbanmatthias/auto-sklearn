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
    apt-get -y install wget git gcc tar
    cd /data
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /data/miniconda
    export PATH=/data/miniconda/bin:$PATH

    conda create -n autonet --yes python=3.6 pip wheel nose gxx_linux-64 gcc_linux-64 swig mkl-service
    source activate autonet
    pip install smac==0.8.0 numpy==1.12.0
    git clone https://github.com/urbanmatthias/auto-sklearn.git
    cd auto-sklearn
    git checkout autonet
    cat requirements.txt | xargs -n 1 -L 1 pip install
    pip install xmltodict requests liac-arff openml xgboost==0.80
    pip install theano==0.9.0
    pip install git+https://github.com/Lasagne/Lasagne@5aadc1b0fa61ad2d27aabbfc6392c5d388e5bba5
    nosetests test -sv
