#!/bin/bash

if [ $# -ne 6 ]; then
    echo -e "\n- usage: $0 <exp_dir/: PATH> <labels_dir/: PATH> <model_iters: INT> <xtr_iter: INT> <clf_type: STR> <ext_ID : STR> [n_train (default: 1000)]\n"
    echo -e "  - Description: This script creates task files that need to be executed on local machine or submitted to a cluster.
    Each task file will:
      Train classifier and predict labels on test data from all the languages.
      Dataset is MLDoc\n"
    echo "  . config_file      : path to config json file corresponding to the model that is used to extract the embeddings."
    echo "  . labels_dir/      : path where all labels live in lang-specific sub_dirs (eg:random_data_splits/)"
    echo "  . model_iters (INT): which trained model to consider. Since filenaming is dependent om this. Usually it is 100"
    echo "  . xtr_iters   (INT): which embeddings to consider. File naming depends on this. Usually it is 50."
    echo "  . clf_type    (STR): gen or disc or both. gen for Gaussian linear classifiers, disc for logisitc regression. both for both"
    echo "  . ext_ID      (STR): results dir extend ID"
    echo -e "\n - Require 5 arguments. Given $#\n"
    exit;
fi


. env.sh

PRE=`pwd`

gen_clf_src=${PRE}"/src/cl_train_and_clf_rcv.py"
disc_clf_src=${PRE}"/src/run_mclr_rcv.py"

langs=("english" "german" "french" "italian" "spanish"  "russian" "japanese" "chinese")


cfg_file=`realpath $1`
labels_dir=`realpath $2`
model_iters=$3
xtr=$4
clf_type=$5

sh_dir=`dirname ${cfg_file}`/tsks
mkdir -p $sh_dir

ext_id=$6

n_train=${7:-1000}

cfg_dir=`dirname ${cfg_file}`

if [ "${clf_type}" = "gen" ] || [ "${clf_type}" = "both" ]; then

    sh_file=${sh_dir}/glc_cpu.tsk
    if [ -f ${sh_file} ]; then rm -f ${sh_file} ; fi
    touch ${sh_file}

    shu_file=${sh_dir}/glcu_cpu.tsk
    if [ -f ${shu_file} ]; then rm -f ${shu_file} ; fi
    touch ${shu_file}

    res_dir=`dirname $cfg_file`/results_${ext_id}/
    mkdir -p ${res_dir}
    echo "*" ${res_dir}

    for src_lang in ${langs[@]}; do
        for k in `seq 1 1 5`; do

            echo "$python ${gen_clf_src} glc ${cfg_file} ${labels_dir}/ ${k} ${src_lang} \
-model_iters ${model_iters} -xtr ${xtr} -ntrain ${n_train} -out_dir ${res_dir} " >> ${sh_file}
            echo "$python ${gen_clf_src} glcu ${cfg_file} ${labels_dir}/ ${k} ${src_lang} \
-trn 5 -model_iters ${model_iters} -xtr ${xtr} -ntrain ${n_train} -out_dir ${res_dir}" >> ${shu_file}
            echo "$python ${gen_clf_src} glcu ${cfg_file} ${labels_dir}/ ${k} ${src_lang} \
-trn 0 -model_iters ${model_iters} -xtr ${xtr} -ntrain ${n_train} -out_dir ${res_dir}" >> ${shu_file}

        done
    done

    wc -l ${sh_file}
    wc -l ${shu_file}

fi

if [ "${clf_type}" = "disc" ] || [ "${clf_type}" = "both" ] ; then

    gpu_sh_file=${sh_dir}/mclr_gpu.tsk
    if [ -f ${gpu_sh_file} ]; then rm -f ${gpu_sh_file} ; fi
    touch ${gpu_sh_file}

    res_dir=`dirname $cfg_file`/results_mclr_${ext_id}/
    mkdir -p ${res_dir}
    echo "*" ${res_dir}

    for src_lang in ${langs[@]}; do
        for k in `seq 1 1 5`; do

            echo "$python ${disc_clf_src} mclr ${cfg_file} ${labels_dir}/ ${k} ${src_lang} \
-model_iters ${model_iters} -xtr ${xtr} -ntrain ${n_train} -out_dir ${res_dir} --safe-gpu" >> ${gpu_sh_file}
            echo "$python ${disc_clf_src} mclru-0 ${cfg_file} ${labels_dir}/ ${k} ${src_lang} \
-model_iters ${model_iters} -xtr ${xtr} -ntrain ${n_train} -out_dir ${res_dir} --safe-gpu" >> ${gpu_sh_file}
            echo "$python ${disc_clf_src} mclru ${cfg_file} ${labels_dir}/ ${k} ${src_lang} \
-model_iters ${model_iters} -xtr ${xtr} -ntrain ${n_train} -out_dir ${res_dir} --safe-gpu" >> ${gpu_sh_file}

        done
    done

    wc -l ${gpu_sh_file}

fi


