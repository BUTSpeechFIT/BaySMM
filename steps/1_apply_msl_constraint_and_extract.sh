#!/bin/bash

# Apply min-sentence length constraint on k-way parallel (incl bi-texts) and
# extract the respective lines/sentences for each text file.

if [ $# -ne 4 ]; then
    echo "- usage: $0 <dataset.list:FILE_PATH> <lang.list:FILE_PATH> <data_dir/: DIR_PATH> <min. sentence length constraint:INT>"
    echo "  . dataset.list : File with list of parallel datasets to use. One per line."
    echo "  . langs.list   : List of language (codes) to use. Space separated."
    echo "  . par_data_dir : Base dir where all the parallel data lives."
    echo "  . msl          : Min sentence length constraint, as used earlier to prepare the data."
    exit;
fi

dset_file=$1
lang_file=$2
data_dir=$(realpath $3)
msl=$4

# set -e


IFS=' ' read -r -a langs < ${lang_file}

# ja->jp, zh->zhs for GlobalVoices
#langs=("de" "en" "es" "fr" "it" "ja" "jp" "ru" "zh" "zhs")
nl=${#langs[@]}

while IFS= read -r line; do

    first_char=${line:0:1}
    if [ "${first_char}" == "#" ]; then
        continue
    fi

    dset=$(echo $line | awk -F" " '{print $1}')
    kway=$(echo $line | awk -F" " '{print $2}')

    echo "${dset} ..  ${kway}"

    if [ "${dset}" == "Europarl" ]; then

        IFS='-' read -r -a kway_langs <<< "${kway}"
        num_l=${#kway_langs[@]}

        # pivot lang for min. sentence length constraint
        pivot="en"

        # create n-way parallel dataset
        python src/get_k_way_parallel_sentences.py --in_dir "${data_dir}"/Europarl/ \
        --dataset Europarl \
        --out_dir "${data_dir}/Europarl/${kway}" \
        --pivot_lang "${pivot}" \
        --other_langs "${kway_langs[@]}"

        python src/select_sent_indices.py \
               ${data_dir}/${dset}/${kway}/Europarl-${num_l}-way.${pivot} \
               ${pivot} \
               ${msl} \
               ${data_dir}/${dset}/${kway}/Europarl-${num_l}-way-${pivot}-msl-${msl}.ixs

        for lang in "${kway_langs[@]}"; do
            python src/extract_lines_based_on_indices.py \
                   ${data_dir}/${dset}/${kway}/Europarl-${num_l}-way.${lang} \
                   ${data_dir}/${dset}/${kway}/Europarl-${num_l}-way-${pivot}-msl-${msl}.ixs \
                   ${data_dir}/${dset}/${kway}/Europarl-${num_l}-way-msl-${msl}.${lang}
        done

    elif [ "${dset}" == "MultiUN" ]; then

        IFS='-' read -r -a kway_langs <<< "${kway}"
        num_l=${#kway_langs[@]}

        pivot="en"

        # create n-way  parallel dataset
        python src/get_k_way_parallel_sentences.py --in_dir "${data_dir}"/MultiUN/ \
            --dataset MultiUN \
            --out_dir "${data_dir}/MultiUN/${kway}" \
            --pivot_lang "${pivot}" \
            --other_langs "${kway_langs[@]}"

        python src/select_sent_indices.py \
               ${data_dir}/${dset}/${kway}/MultiUN-${num_l}-way.${pivot} \
               ${pivot} \
               ${msl} \
               ${dset}/${kway}/MultiUN-${num_l}-way-${pivot}-msl-${msl}.ixs

        for lang in "${kway_langs[@]}"; do
            python src/extract_lines_based_on_indices.py \
                   ${data_dir}/${dset}/${kway}/MultiUN-${num_l}-way.${lang} \
                   ${data_dir}/${dset}/${kway}/MultiUN-${num_l}-way-${pivot}-msl-${msl}.ixs \
                   ${data_dir}/${dset}/${kway}/MultiUN-${num_l}-way-msl-${msl}.${lang}
        done

    elif [ "${dset}" == "UN" ]; then

        cmn="v1.0.6way"

        pivot="en"

        if [ -d "${data_dir}/${dset}/${kway}" ]; then

            python src/select_sent_indices.py \
                ${data_dir}/${dset}/${kway}/UN${cmn}.${pivot} \
                ${pivot} \
                ${msl} \
                ${data_dir}/${dset}/${kway}/UN${cmn}-${pivot}-msl-${msl}.ixs

            for lang in en fr es ru zh; do
                python src/extract_lines_based_on_indices.py \
                    ${data_dir}/${dset}/${kway}/UN${cmn}.${lang} \
                    ${data_dir}/${dset}/${kway}/UN${cmn}-${pivot}-msl-${msl}.ixs \
                    ${data_dir}/${dset}/${kway}/UN${cmn}-msl-${msl}.${lang}

                head -1000000 ${data_dir}/${dset}/${kway}/UN${cmn}-msl-${msl}.${lang} > ${data_dir}/${dset}/${kway}/UN${cmn}-1mil-msl-${msl}.${lang}
            done
        fi

    else

        for ((i=0 ; i<${nl} ; i++)); do
            for ((j=$((i+1)) ; j<${nl} ; j++)); do

                lang1=${langs[i]}
                lang2=${langs[j]}

                if [ ! -z ${kway} ]; then
                    if [ "${lang1}-${lang2}" != "${kway}" ]; then
                        continue
                    fi
                    echo "- ${dset} ${kway}"
                    # in_file1=${data_dir}/${dset}/${kway}/${dset}.${lang1}-${lang2}.${lang1}
                    # in_file2=${data_dir}/${dset}/${kway}/${dset}.${lang1}-${lang2}.${lang2}

                    # msl_file=${data_dir}/${dset}/${kway}/${dset}.${lang1}-${lang2}.${lang1}-msl-${msl}.ixs

                    # out_file1=${data_dir}/${dset}/${kway}/${dset}.${lang1}-${lang2}-msl-${msl}.${lang1}
                    # out_file2=${data_dir}/${dset}/${kway}/${dset}.${lang1}-${lang2}-msl-${msl}.${lang2}

                fi

                # else

                in_file1=${data_dir}/${dset}/${lang1}-${lang2}/${dset}.${lang1}-${lang2}.${lang1}
                in_file2=${data_dir}/${dset}/${lang1}-${lang2}/${dset}.${lang1}-${lang2}.${lang2}

                msl_file=${data_dir}/${dset}/${lang1}-${lang2}/${dset}.${lang1}-${lang2}.${lang1}-msl-${msl}.ixs

                out_file1=${data_dir}/${dset}/${lang1}-${lang2}/${dset}.${lang1}-${lang2}-msl-${msl}.${lang1}
                out_file2=${data_dir}/${dset}/${lang1}-${lang2}/${dset}.${lang1}-${lang2}-msl-${msl}.${lang2}
                # fi

                # printf ${in_file1}"\n"
                # printf ${in_file2}"\n"

                if [ -f ${in_file1} ]; then



                    python src/select_sent_indices.py ${in_file1} ${lang1} ${msl} ${msl_file}


                    python src/extract_lines_based_on_indices.py ${in_file1} ${msl_file} ${out_file1}
                    python src/extract_lines_based_on_indices.py ${in_file2} ${msl_file} ${out_file2}

                fi

            done
        done

    fi

done < ${dset_file}
