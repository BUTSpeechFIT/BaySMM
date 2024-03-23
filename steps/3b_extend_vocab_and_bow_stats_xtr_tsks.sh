#!/usr/bin/env bash

if [ $# -ne 8 ]; then
    echo -e "\n - usage: $0"
    echo -e "
 - Description: This is needed when extending an already trained model to newer languages.
                Atleast one pivot language from the existing trained model is required.\n"
    echo -e " - Arguments:"
    echo "  . langs.list      : File containing list of language (codes) to use. Space separated."
    echo "  . pivot.list      : File containing list of pivot language(s) to use. Space separated."
    echo "  . ext_ID          : Unique ID (string) for this extended training set. The same ext_ID will be required in later steps 4b, 5b and 6b."
    echo "  . min-doc-freq    : Min doc freq constraint while building the vocabulary for newer languages (INT)."
    echo "  . train_flist_dir/: Dir path to where lang.flist files were saved in the prev step 2"
    echo "  . xtr_flist_dir/  : Dir path to where lang.flist files were saved. These should correspond to all splits from \
either MLDoc or INA or any other corpus for which you intend to extract the embeddings."
    # echo "  . xtr_dset_name   : rcv (for MLDoc) or ina or INA"
    echo "  . bow_stats_dir/  : Existing BoW stats dir to save the doc-by-word stats, which will be used for extending the training to newer languages."
    echo "  . tsk_dir/        : Dir to save the task files that needs to be executed or submitted to a cluster"
    echo -e "\n - Require 8 arguments. Given $#"
    echo
    exit;
fi

PRE=`pwd`
xtr_src_file=${PRE}/src/extract_features_for_test_data.py
src_file=${PRE}/src/feature_xtr_from_text_spm_v2.py


. env.sh

echo $python

lang_file=`realpath $1`
pivot_file=`realpath $2`

ext_id=$3

input_type="text"
tokenizer="normal"
ng=1
mdf=$4

flist_dir=`realpath $5`
xtr_flist_dir=`realpath $6`

base_bow_dir=`realpath $7`
out_dir=${base_bow_dir}/${ext_id}
mkdir -p ${out_dir}

echo "- Out BoW dir for new bitexts and langs: ${out_dir}"

mkdir -p $8
tsk_dir=`realpath $8`

IFS=' ' read -r -a langs < ${lang_file}
IFS=' ' read -r -a pivots < ${pivot_file}

echo "- Languages    : ${langs[@]}"
echo "- Pivot lang(s): ${pivots[@]}"


nl=${#langs[@]}

for ((i=0 ; i<${nl} ; i++)); do

    lid=${langs[$i]}
    lang=${lid}

    if [ "${lid}" == "zh_cn" ]; then
        lang="zh"
    fi

    echo "- ${lid}"

    input_flist=${flist_dir}/${lid}.flist

    sh_file=${tsk_dir}/${lid}_bow_xtr.sh
    log_file=${tsk_dir}/${lid}_bow_xtr.log
    echo "#/usr/bin/env bash" > ${sh_file}

    if [ -f ${input_flist} ]; then

        echo "   - Training flist: ${input_flist}"
        # check if the lang is in pivot langs
        flag=0
        for pl in ${pivots[@]}; do
            if [ "${pl}" == "${lang}" ]; then
               flag=1
               break;
            fi
        done

        if [ "${flag}" == 1 ]; then
            # pivot lang, we will use existing CountVectorizer and vocabulary
            # and extract BoW stats for the parallel bitext

            cvect_pkl_file=`find ${base_bow_dir}/ -name "${lang}_cvect_*.pkl"`
            if [ -f ${cvect_pkl_file} ]; then
                echo "   - Pivot language. Will use existing CountVectorizer : ${cvect_pkl_file}"
                echo "(( ${python} ${xtr_src_file} ${lang} ${input_flist} flist \
plain ${cvect_pkl_file} ${out_dir}/${lang}/parallel/ ) 2>&1 ) >> ${log_file}" >> ${sh_file}
            else
                echo "ERROR: Cannot find ${lang} CountVectorizer pkl file: ${cvect_pkl}"
                echo "Make sure that the out_dir points to the same dir where the ${lang} CountVectorizer exists"
                exit;
            fi

        else

            xtr_flist=${xtr_flist_dir}/${lang}.flist
            if [ -f ${xtr_flist} ]; then
                echo "   - xtr flist: ${xtr_flist}"

                # We will use all the XX-YY parallel data for each language
                # and build vocabulary
                # Then extract BoW stats for these parallel texts and
                # also for MLDoc/INA data (we don't learn vocab from MLDoc, INA)
                # MLDoc is rcv, INA is IndicNLP-News-Articles Classification dataset
                echo "(( ${python} ${src_file} \
${input_flist} flist ${lang} ${out_dir}/ \
-extract_list ${input_flist} ${xtr_flist} \
-xtr_out_dir ${out_dir}/${lid}/parallel/ ${out_dir}/${lid}/ina/ \
-ng $ng -mdf $mdf ) 2>&1 ) > ${log_file}" >> ${sh_file}

                # We create a json file with BoW stats npz files
                # corresponding to the MLDoc set
                # This json file will be needed to extract embeddings
                # from Multi BaySMM
                echo "(( ${python} ${PRE}/src/create_xtr_json.py ${lang} \
 -in_dir ${out_dir}/${lid}/ina/ \
 -out_file ${out_dir}/xtr_npz_${ext_id}.json ) 2>&1 ) >> ${log_file}" >> ${sh_file}

            else
                echo "- Warning: ${xtr_flist} not found."
            fi

        fi

    else

        echo "${input_flist} NOT FOUND."
        exit;

    fi


done

chmod a+x ${tsk_dir}/*.sh

tsk_file=${tsk_dir}/bow_xtr.tsk
find ${tsk_dir}/ -name "*.sh" > ${tsk_file}

echo -e "\n\nRun ${tsk_file}"
echo "  on local machine using \"source ${tsk_file}\""
echo "  or submit it to a cluster."
