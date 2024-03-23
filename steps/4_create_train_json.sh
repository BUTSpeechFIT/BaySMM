#!/bin/bash

# Create parallel_train_data.json that contains
# K-way parallel BoW stats (npz format) and the vocabulary (json) files
# Each parallel dataset is given a unique number

if [ $# -ne 3 ]; then
    echo "- usage: $0 <dataset.list:FILE_PATH> <lang.list:FILE_PATH> <bow_stats_base_dir/:DIR_PATH>"
    exit;
fi

dset_file=$1
lang_file=$2
in_base=$(realpath $3)

out_file=${in_base}/parallel_train_data.json
out_vocab_file=${in_base}/lang_vocab.json

if [ -f ${out_file} ]; then
    echo ${out_file}" already exists."
    exit;
fi

IFS=' ' read -r -a langs < "${lang_file}"
nl=${#langs[@]}

echo "- Languages: " "${langs[@]}"

set -e

# create lang vocab json file
for lang in "${langs[@]}"; do
    if [ "${lang}" == "zh_cn" ]; then
        lang="zh"
    fi

    lang_vocab_json_file=$(find ${in_base}/ -type f -name "${lang}_vocab*.json")
    if [ -z ${lang_vocab_json_file} ]; then
        echo "Error: Cannot find ${lang}_vocab_*.json for language: ${lang} in dir ${in_base}/"
        exit;
    fi
    python src/create_lang_vocab_json.py \
            ${lang_vocab_json_file}\
            ${lang} \
            ${out_vocab_file}
done

num=1
# create parallel dataset num to npz mapping json file
while IFS= read -r line; do

    first_char=${line:0:1}
    if [ "${first_char}" == "#" ]; then
        continue
    fi

    dset=$(echo $line | cut -d' ' -f1)
    kway=$(echo $line | cut -d' ' -f2)

    echo "  ${dset} - "

    if [ "${dset}" == "Europarl" ] || [ "${dset}" == "UN" ] || [ "${dset}" == "MultiUN" ]; then

        for lang in "${langs[@]}"; do
            n_chunks=$(find ${in_base}/${lang}/parallel/ -type f -name "${dset}*chunk*.npz" | wc -l)
            if [ ${n_chunks} -gt 0 ]; then
                break
            fi
        done

        if [ ${n_chunks} -gt 0 ]; then
            for c in $(seq 1 ${n_chunks}); do
                for lang in "${langs[@]}"; do
                    npz_file=$(find ${in_base}/${lang}/parallel/ -type f -name "${dset}*chunk_${c}.npz")
                    if [ ! -z ${npz_file} ]; then
                        echo " ${lang} | chunk ${c} "
                        python src/create_train_json.py ${num} ${lang} ${npz_file} ${out_file}
                    fi
                done
                # increment the parallel dataset number
                num=$((num+1))
            done
        else
            for lang in "${langs[@]}"; do
                npz_file=$(find ${in_base}/${lang}/parallel/ -type f -name "${dset}*.npz")
                if [ ! -z ${npz_file} ]; then
                    python src/create_train_json.py ${num} ${lang} ${npz_file} ${out_file}
                fi
            done
            # increment the parallel dataset number
            num=$((num+1))
        fi

    else
        # these are the bitext pairs and each one will have a different dataset num
        for ((i=0 ; i<${nl} ; i++)); do
            lang1=${langs[$i]}

            # if [ "${lang1}" == "zh_cn" ]; then
            #    lang1="zh"
            # fi

            for ((j=$((i+1)) ; j<${nl} ; j++)); do
                lang2=${langs[$j]}
                # if [ "${lang2}" == "zh_cn" ]; then
                #    lang2="zh"
                #fi

                pair=${lang1}-${lang2}

                npz_file1=$(find ${in_base}/${lang1}/parallel/ -type f -name "${dset}.${pair}*.npz")
                npz_file2=$(find ${in_base}/${lang2}/parallel/ -type f -name "${dset}.${pair}*.npz")

                if [ ! -z ${npz_file1} ] && [ ! -z ${npz_file2} ]; then

                    if [ "${lang1}" == "zh_cn" ]; then
                        lang1="zh"
                    fi

                    if [ "${lang2}" == "zh_cn" ]; then
                        lang2="zh"
                    fi

                    python src/create_train_json.py ${num} ${lang1} ${npz_file1} ${out_file}
                    python src/create_train_json.py ${num} ${lang2} ${npz_file2} ${out_file}

                    # increment the parallel dataset number
                    num=$((num+1))
                fi
            done

        done

    fi


done < ${dset_file}
