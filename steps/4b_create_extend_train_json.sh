#!/bin/bash

# Create parallel_extend_train_data.json that can be used to extend
# an existing model to newer languages. The json file contains
# K-way parallel BoW stats (npz format) and the vocabulary (json) files
# Each parallel dataset is given a unique number

if [ $# -ne 5 ]; then
    echo "- usage: $0 "
    echo -e "
   Description: Create parallel_extend_train_data.json that can be used to extend
                an existing model to newer languages. The json file contains
                K-way parallel BoW stats (npz format) and the vocabulary (json) files.
                Each parallel dataset is given a unique number.
     "
    echo "  . dataset.list       : path dataset.list file that contains the list of datasets used to extend the model"
    echo -e "  . lang.list          : path to langs.list file that contains the list of languages to
                         which the model is to be extended. It should also contain the pivot language from
                         the list of already trained languages."
    echo -e "  . pivot_lang.list    : File with list (single line space separated) of pivot languages,
                         whose params will not be updated."
    # echo "  . pivot_bow_stats_dir: Dir path where the bow stats (npz) files were saved from the pivot previous step 3b"
    echo -e "  . new_bow_stats_dir  : Dir path where the bow stats (npz) files were saved from the previous step 3b
                         The bow dir for the pivot lang is expected to be one directory up (since it was created in step 3b)"
    echo "  . ext_ID             : Unique ID (string) for this extended training set."
    echo -e "\n Require 5 args. Given $#\n"
    exit;
fi

. env.sh

dset_file=$1
lang_file=$2
pivot_file=$3

new_bow_dir=`realpath $4`
pivot_bow_dir=${new_bow_dir}/../

ext_id=$5

out_file=${new_bow_dir}/extended_parallel_train_data_${ext_id}.json
out_vocab_file=${new_bow_dir}/lang_vocab.json

if [ -f ${out_file} ]; then
    echo ${out_file}" already exists."
    exit;
fi

IFS=' ' read -r -a langs < ${lang_file}
nl=${#langs[@]}
echo "- Languages: ${langs[@]}"

IFS=' ' read -r -a pivot < ${pivot_file}
echo "- Pivot lang:"${pivot[@]}

set -e

# create lang vocab json file
for lang in ${langs[@]}; do
    if [ "${lang}" == "zh_cn" ]; then
        lang="zh"
    fi

    is_pivot=0
    for pl in ${pivot[@]}; do
        if [ "${lang}" == "${pl}" ]; then
            is_pivot=1
            break
        fi
    done

    if [ ${is_pivot} == 1 ]; then
        lang_vocab_json_file=`find ${pivot_bow_dir}/ -type f -name "${lang}_vocab_*.json"`
    else
        lang_vocab_json_file=`find ${new_bow_dir}/ -type f -name "${lang}_vocab_*.json"`
    fi
    if [ ! -f ${lang_vocab_json_file} ]; then
        echo "Error: Cannot find ${lang}_vocab_*.json for language: ${lang} in dir ${in_base}/"
        exit;
    fi

    $python src/create_lang_vocab_json.py \
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

    dset=`echo $line | cut -d' ' -f1`
    kway=`echo $line | cut -d' ' -f2`

    echo "  ${dset} - "

    # These are L-way parallel datasets where L > 2
    if [ "${dset}" == "Europarl" ] || [ "${dset}" == "UN" ] || [ "${dset}" == "MultiUN" ]; then

        for lang in ${langs[@]}; do
            n_chunks=`find ${new_bow_dir}/${lang}/parallel/ -type f -name "${dset}*chunk*.npz" | wc -l`
            if [ ${n_chunks} -gt 0 ]; then
                break
            fi
        done

        if [ ${n_chunks} -gt 0 ]; then
            for c in `seq 1 ${n_chunks}`; do
                for lang in ${langs[@]}; do
                    npz_file=`find ${new_bow_dir}/${lang}/parallel/ -type f -name "${dset}*chunk_${c}.npz"`
                    if [ ! -z ${npz_file} ]; then
                        # echo " ${lang} | chunk ${c} "

                        trainable="--trainable"
                        # check if lang is a pivot lang
                        for pl in ${pivot[@]}; do
                            if [ "${pl}" == "${lang}" ]; then
                                trainable=""
                            fi
                        done

                        $python src/create_train_json.py ${num} ${lang} ${npz_file} ${out_file} ${trainable}
                    fi
                done
                # increment the parallel dataset number
                num=$((num+1))
            done

        else

            for lang in ${langs[@]}; do
                npz_file=`find ${new_bow_dir}/${lang}/parallel/ -type f -name "${dset}*.npz"`
                if [ ! -z ${npz_file} ]; then
                    # echo " ${lang} |  "

                    trainable="--trainable"
                    # check if lang is a pivot lang
                    for pl in ${pivot[@]}; do
                        if [ "${pl}" == "${lang}" ]; then
                            trainable=""
                        fi
                    done

                    $python src/create_train_json.py ${num} ${lang} ${npz_file} ${out_file} ${trainable}
                fi
            done
            # increment the parallel dataset number
            num=$((num+1))

        fi


    else
        # These are the bitext pairs and each one will have a different dataset num
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


                npz_file1=`find ${new_bow_dir}/${lang1}/parallel/ -type f -name "${dset}.${pair}*.npz"`
                npz_file2=`find ${new_bow_dir}/${lang2}/parallel/ -type f -name "${dset}.${pair}*.npz"`

                if [ ! -z ${npz_file1} ] && [ ! -z ${npz_file2} ]; then

                    echo "    - ${pair}"

                    if [ "${lang1}" == "zh_cn" ]; then
                        lang1="zh"
                    fi

                    if [ "${lang2}" == "zh_cn" ]; then
                        lang2="zh"
                    fi

                    trainable1="--trainable"
                    trainable2="--trainable"

                    # check if lang1 or lang2 is a pivot lang
                    for pl in ${pivot[@]}; do
                        if [ "${pl}" == "${lang1}" ]; then
                            trainable1=""
                        fi
                        if [ "${pl}" == "${lang2}" ]; then
                            trainable2=""
                        fi
                    done

                    $python src/create_train_json.py ${num} ${lang1} ${npz_file1} ${out_file} ${trainable1}
                    $python src/create_train_json.py ${num} ${lang2} ${npz_file2} ${out_file} ${trainable2}

                    # increment the parallel dataset number
                    num=$((num+1))
                fi
            done

        done

    fi


done < ${dset_file}
