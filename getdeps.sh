#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source ${DIR}/envvars.sh
mkdir -p ${INSTALL_DIR}
echo "INSTALL_DIR: ${INSTALL_DIR}"

function download() {
    url=$1
    dst_file=$2
    if [ ! -f $dst_file ]
    then
        wget $url
    else
        echo "$dst_file already exists. Not downloading again."    
    fi
}

function unpack() {
    tarfile=$1
    dst_dir=$2
    tar xf $tarfile
}

function copy_targets() {
    targets_list=$1
    src_dir=$2
    dst_dir=$3
    mkdir -p ${dst_dir}
    for tgt in $targets_list
    do
        cp ${src_dir}/${tgt} ${dst_dir}
    done
}

# OpenBLAS Serial
OPENBLAS_VERSION=0.2.19
OPENBLAS_FILE=v${OPENBLAS_VERSION}.tar.gz
OPENBLAS_URL="https://github.com/xianyi/OpenBLAS/archive/${OPENBLAS_FILE}"
OPENBLAS_DIR=OpenBLAS-${OPENBLAS_VERSION}

download ${OPENBLAS_URL} ${OPENBLAS_FILE}
unpack ${OPENBLAS_FILE} ${OPENBLAS_DIR}
cd ${OPENBLAS_DIR}
if [ ! -e libopenblas.so ]
then
    make USE_THREAD=0
    make PREFIX=${INSTALL_DIR} install
else
    echo "${OPENBLAS_DIR}/libopenblas.so already exists. Not rebuilding OpenBlas."
fi
cd ..

# Cereal
CEREAL_VERSION=1.2.2
CEREAL_FILE=v${CEREAL_VERSION}.tar.gz
CEREAL_URL="https://github.com/USCiLab/cereal/archive/${CEREAL_FILE}"
CEREAL_DIR=cereal-${CEREAL_VERSION}

download ${CEREAL_URL} ${CEREAL_FILE}
unpack ${CEREAL_FILE} ${CEREAL_DIR}
mkdir -p ${INSTALL_DIR}/include
cp -a ${CEREAL_DIR}/include/cereal ${INSTALL_DIR}/include

