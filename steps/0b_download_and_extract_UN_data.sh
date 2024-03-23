#!/bin/bash

if [ $# -ne 1 ]; then
    echo "$0 <data_dir/>"
    exit;
fi

data="${1}/UN/"
mkdir -p "${data}"

wget -q -P ${data} -nc https://conferences.unite.un.org/uncorpus/Home/DownloadFile?filename=UNv1.0.6way.tar.gz.00
wget -q -P ${data} -nc https://conferences.unite.un.org/uncorpus/Home/DownloadFile?filename=UNv1.0.6way.tar.gz.01
wget -q -P ${data} -nc https://conferences.unite.un.org/uncorpus/Home/DownloadFile?filename=UNv1.0.6way.tar.gz.02
wget -q -P ${data} -nc https://conferences.unite.un.org/uncorpus/Home/DownloadFile?filename=UNv1.0.6way.tar.gz.03

if [ -f ${data}/UNv1.0.6way.tar.gz.03 ]; then
    cat ${data}/UNv1.0.6way.tar.gz.* > ${data}/UNv1.0.6way.tar.gz
    tar -xzf "${data}/UNv1.0.6way.tar.gz"
else
    echo "Could not download UN corpus. Do it manually."
fi


