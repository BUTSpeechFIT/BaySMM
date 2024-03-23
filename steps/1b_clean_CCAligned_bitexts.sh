#!/bin/bash

if [ $# -ne 4 ]; then
    echo "- usage: $0 <dset_name> <lang.list:FILE_PATH> <data_dir/: DIR_PATH> <min. sentence length constraint:INT>"
    echo "  . dset_name    : Dataset name, a common prefix to all files OPUS format."
    echo "  . langs.list   : List of language (codes) to use. Space separated."
    echo "  . par_data_dir : A particular parallel dir where all data from different langauges live."
    echo "  . msl          : Min sentence length constraint, as used earlier to prepare the data."
    exit;
fi

. env.sh

PRE=`pwd`

dset=$1
lang_file=$2
data_dir=`realpath $3`
msl=$4

IFS=' ' read -r -a langs < ${lang_file}

# ja->jp, zh->zhs for GlobalVoices
#langs=("de" "en" "es" "fr" "it" "ja" "jp" "ru" "zh" "zhs")
nl=${#langs[@]}

set -e

for ((i=0 ; i<${nl} ; i++)); do
    for ((j=$((i+1)) ; j<${nl} ; j++)); do

        lang1=${langs[i]}
        lang2=${langs[j]}

        in_file1=${data_dir}/${lang1}-${lang2}/${dset}.${lang1}-${lang2}-msl-${msl}.${lang1}
        in_file2=${data_dir}/${lang1}-${lang2}/${dset}.${lang1}-${lang2}-msl-${msl}.${lang2}

        if [ -f ${in_file1} ]; then
            echo "- ${lang1}-${lang2} .."
            if [ "${lang1}" == "en" ]; then
                python ${PRE}/src/clean_bitexts.py ${in_file1} ${in_file2} --replace
            elif [ "${lang2}" == "en" ]; then
                python ${PRE}/src/clean_bitexts.py ${in_file2} ${in_file1} --replace
            else
                echo "skipping"
            fi
        fi

    done
done
