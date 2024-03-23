#!/bin/bash

if [ $# -ne 5 ]; then
    echo ""
    echo " - USAGE: $0 <old_config.json>  <extend_parallel_train_data.json> <new_xtr_data.json> <ext_ID> <tsks_dir/>"

    echo -e "
  Description: Creates a task file that will extend an existing model to newer languages.
               The task file can be executed on a local machine with GPU or submitted to a cluster.\n"
    echo -e "  . runs_dir                         :
       Path to the old (existing) exp_base dir runs/ "
    echo -e "  . extend_parallel_train_data.json :
       Path to the json file containing paths to parallel exts for the new languages.
       This was created in the previous step 4b"
    echo -e "  . new_xtr_data.json :
       Path to json file containing paths to the npz files for which embeddings will be extracted.
       This was created in the previous step 4b"
    echo "  . ext_ID   : Unique ID (string) for this extended training set (same as in prev step 4b)."
    echo "  . tsk_dir/ : Dir to save the task files that needs to be executed or submitted to a cluster"
    echo -e "\n Require 5 args. Given $# \n"
    exit
fi

set -e


. env.sh


PRE=`pwd`

echo $python

exp_dir=`realpath $1`
train_f=`realpath $2`
bow_dir=`dirname ${train_f}`
lang_vocab_f=${bow_dir}/lang_vocab.json
xtr_f=`realpath $3`

ext_id=$4
sfx=$4

mkdir -p $5
sh_dir=`realpath $5`



R=8
vp=1e+01
lw=1e+01
trn=100
save=20
Ks=(256)
rt='l2'
#lts=(5e-03 5e-04)
lts=(5e-03)  # best hyper-param on dev same-same language acc
xtr=50
nth=50
eta=0.005
bsize=512

# sleep timer
# st=1

for K in ${Ks[@]}; do

    for lt in ${lts[@]}; do

        sh_file=${sh_dir}/mbay_${lt}_${K}.sh
        log_file=${sh_dir}/mbay_${lt}_${K}.log
        touch $log_file
        echo "${sh_file}"
        echo "#!/bin/bash" > ${sh_file}

        mbase=multi_r_${R}_vp_${vp}_lw_${lw}_l2_${lt}_${K}_adam
        cfg_file=${exp_dir}/${mbase}/config.json
        final_model=${exp_dir}/${mbase}/models/${ext_id}/model_T${trn}.pt

        if [ ! -f ${final_model} ]; then

            echo -e "(( ${python} ${PRE}/src/run_mbay.py extend \\
 ${train_f} \\
 ${lang_vocab_f} \\
 ${cfg_file} \\
 ${ext_id} -bsize ${bsize} --safe-gpu \\
) 2>&1 ) >> ${log_file} " >> ${sh_file}
        fi


        ext_cfg_file=${exp_dir}/${mbase}/config_${ext_id}.json
        echo -e "(( ${python} ${PRE}/src/run_mbay.py extract \\
 ${xtr_f} \\
 ${ext_cfg_file} \\
 -xtr ${xtr} -nth ${nth} -bsize ${bsize} --safe-gpu \\
) 2>&1 ) >> ${log_file}" >> ${sh_file}


    done
done


tsk_file=${sh_dir}/mbay-${sfx}.tsk
chmod a+x ${sh_dir}/*.sh
find ${sh_dir}/ -name "*.sh" > ${tsk_file}


echo -e "\n- Run ${tsk_file} or submit to a cluster.\n"
