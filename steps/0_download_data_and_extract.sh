#!/bin/bash

if [ $# -ne 3 ]; then

    echo "- usage: $0 <dataset.list:FILE_PATH> <lang.list: FILE_PATH> <download_base_dir/:DIR_PATH>"
    echo "  dataset.list is one dataset name per line"
    echo "  lang.list is two letter ISO code separated by space"
    exit

fi

# langs=("af" "am" "ar" "arq" "as" "ast" "az" "be" "bg" "bi" "bn" "bo" "bs" \
#             "ca" "ceb" "cs" "da" "de" "dz" "el" "en" "eo" "es" "et" "eu" \
#             "fa" "fi" "fil" "fr" "fr_ca" "ga" "gl" "gu" \
#             "ha" "he" "hi" "hr" "ht" "hu" "hup" "hy" "id" "ig" "inh" "is" "it" \
#             "ja" "ka" "kk" "km" "kn" "ko" "ku" "ky" "la" "lb" "lo" "lt" "ltg" "lv" \
#             "mg" "mk" "ml" "mn" "mr" "ms" "mt" "my" "nb" "ne" "nl" "nn" \
#             "oc" "pa" "pl" "ps" "pt" "pt_br" "ro" "ru" \
#             "sh" "si" "sk" "sl" "so" "sq" "sr" "srp" "sv" "sw" "szl" \
#             "ta" "te" "tg" "th" "tk" "tl" "tlh" "tr" "tt" "ug" "uk" "ur" "uz" \
#             "vi" "zh" "zh_cn" "zh_tw" )
#langs=("de" "en" "es" "fr" "it" "ja" "jp" "ru" "zh" "zhs")


# The opus.nlpl.eu services changed the common download URL to the following.
# This was tested on Jan 10 2022
url_prefix="https://object.pouta.csc.fi/OPUS"

dset_file=$1
lang_file=$2

mkdir -p ${3}
download_dir=$(realpath ${3})

IFS=' ' read -r -a langs < "${lang_file}"
echo "- Languages:" "${langs[@]}"


while IFS= read -r line; do

    first_char=${line:0:1}
    if [ "${first_char}" == "#" ]; then
        continue
    fi

    dset=$(echo $line | cut -d' ' -f1)
    kway=$(echo $line | cut -d' ' -f2)

    case ${dset} in
        "Europarl")
            ver="v8";;
        "MultiUN")
            ver="v1";;
        "GlobalVoices")
            ver="v2018q4";;
        "News-Commentary")
            ver="v16";;
        "TED2020")
            ver="v1";;
        "ECB")
            ver="v1";;
        "DGT")
            ver="v2019";;
        "CCAligned")
            ver="v1";;
        "UN")

          ./0b_download_and_extract_UN_data.sh

            continue;;
        "Wiki")
            echo "Need to download Wiki from a different link"
            continue;;
        "indic2indic")
            echo "Need to download indic2indic from different link"
            continue;;
        "pib-v1.3")
            echo "Need to download pib-v1.3 from different link"
            continue;;
        "mkb")
            echo "Need to download mkb from different link"
            continue;;
        "Tatoeba")
            ver="v2021-07-22";;
    esac

    set +e

    # zip files are downloaded here
    zip_dir=${download_dir}/${dset}/zip
    mkdir -p ${zip_dir}

    nl=${#langs[@]}
    echo "- Downloading bitext pairs from ${dset}-${ver} to ${zip_dir}"

    for ((i=0 ; i<${nl} ; i++)); do
        for ((j=0 ; j<${nl} ; j++)); do

            lang1=${langs[i]}
            lang2=${langs[j]}

            # url=https://opus.nlpl.eu/download.php?f=${dset}/${ver}/moses/${lang1}-${lang2}.txt.zip
            url=${url_prefix}-${dset}/${ver}/moses/${lang1}-${lang2}.txt.zip

            http_status=$( wget -S --spider "${url}" 2>&1 )
            exit_status=$?
            http_status=$( awk '/HTTP\//{print $2 }' <<<"${http_status}" | tail -n 1 )

            # echo "  - url: ${url}  | http_stats=us: $http_status"

            if [ ${http_status} != 404 ]; then
                wget -q -P ${zip_dir}/ -nc ${url}
                zip_f=${zip_dir}/${lang1}-${lang2}.txt.zip

                if [ -f ${zip_f} ]; then
                    xtr_dir=${download_dir}/${dset}/${lang1}-${lang2}/
                    echo "  - ${lang1}-${lang2} OK !"
                    # unzip -q -n -d ${xtr_dir}/ ${zip_dir}/download.php?f=${dset}%2F${ver}%2Fmoses%2F${lang1}-${lang2}.txt.zip
                    unzip -q -n -d ${xtr_dir} ${zip_f}

                fi

            else
                # echo "  - ${lang1}-${lang2} ${http_status} ${url}"
                echo -n ""
            fi

        done

    done

done < ${dset_file}
