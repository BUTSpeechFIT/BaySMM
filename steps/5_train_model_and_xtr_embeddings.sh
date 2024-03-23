#!/bin/bash

if [ $# -lt 1 ]; then
    echo ""
    echo " - USAGE: $0 <bow_stats_dir/> [sfx for tsk: (default:6L)]"
    echo ""
    exit
fi

set -e


. env.sh

PRE=$(pwd)


bow_dir=$(realpath $1)
train_f=${bow_dir}/parallel_train_data.json
lang_vocab_f=${bow_dir}/lang_vocab.json
xtr_f=${bow_dir}/xtr_npz.json

exp_dir=${bow_dir}/runs
mkdir -p ${exp_dir}

sfx=${2:-6L}

sh_dir=${bow_dir}/tsks/mbay_train_${sfx}/
mkdir -p ${sh_dir}

R=8
vp=1e+01
lw=1e+01
trn=100
save=20
Ks=(256)
rt='l2'
#lts=(5e-02 5e-03 5e-04)
lts=(5e-03)  # best hyper-param on dev same-same language acc
xtr=50
nth=50
eta=0.005
bsize=512

# sleep timer
# st=1

for K in "${Ks[@]}"; do

    for lt in "${lts[@]}"; do

        sh_file=${sh_dir}/mbay_${lt}_${K}.sh
        log_file=${sh_dir}/mbay_${lt}_${K}.log
        touch $log_file
        echo "${sh_file}"
        echo "#!/bin/bash" > ${sh_file}

        mbase=multi_r_${R}_vp_${vp}_lw_${lw}_l2_${lt}_${K}_adam
        cfg_file=${exp_dir}/${mbase}/config.json
        final_model=${exp_dir}/${mbase}/models/model_T${trn}.pt

        if [ ! -f ${final_model} ]; then

            # echo "sleep ${st}" >> ${sh_file}
            # echo "export \`gpus\`" >> ${sh_file}
            echo "(( ${python} ${PRE}/src/run_mbay.py train ${train_f} ${lang_vocab_f} ${exp_dir}/ \
-K $K -lt $lt -trn $trn -save $save -var_p 1e+01 -lw 1e+01 -eta $eta \
-bsize ${bsize} --safe-gpu ) 2>&1 ) >> ${log_file} " >> ${sh_file}
        fi

        # echo "sleep ${st}" >> ${sh_file}
        # echo "export \`gpus\`" >> ${sh_file}
        echo "(( ${python} ${PRE}/src/run_mbay.py extract ${xtr_f} ${cfg_file} \
-xtr ${xtr} -nth ${nth} -bsize ${bsize} --safe-gpu ) 2>&1 ) >> ${log_file}" >> ${sh_file}

        # st=$((st+10))

    done
done

tsk_file=${sh_dir}/mbay-${sfx}.tsk
chmod a+x ${sh_dir}/*.sh
find ${sh_dir}/ -name "*.sh" > ${tsk_file}

echo "See and run the above script"