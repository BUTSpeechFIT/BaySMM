#!/bin/bash

if [ $# -ne 3 ] ; then
    echo '- usage $0 <dir_with_random_data_splits/> <flist_dir/> <dset_name: [mldoc, ina]>'
    exit;
fi

data_dir=`realpath $1`
mkdir -pv $2
flist_dir=`realpath $2`

dset_name=$3

if [ "${dset_name}" == "mldoc" ]; then
    langs=("english" "german" "french" "italian" "spanish" "russian" "japanese" "chinese")
elif [ "${dset_name}" == "ina" ]; then
    langs=("bn" "gu" "kn" "ml" "mr" "or" "pa" "ta" "te")
else
    echo "dset_name: ${dset_name} should be one from mldoc or ina"
    exit;
fi

for lang in ${langs[@]}; do

    find ${data_dir}/${lang}/ -name "${lang}_*split*.txt" > ${flist_dir}/${lang}_all_splits.flist
    wc -l ${flist_dir}/${lang}_all_splits.flist

done
