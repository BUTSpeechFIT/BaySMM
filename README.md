# MBay: A Bayesian Multilingual Document Model for Zero-shot Topic Identification and Discovery

- The original implementation of BaySMM is now moved to [legacy](legacy/)
- The multilingual version also supports monolingual training.

### About

* The paper is available on [arXiv](https://arxiv.org/abs/2007.01359v3)

* See [INSTALL.md](INSTALL.md) for requirements and installation instructions.

* `steps/` - Contains recipes for downloading parallel data from OPUS to train MBay models and downstream classifiers
* `maby/` - Source code for the model definition, training utils, etc.
* `scripts/` - Source code for data preparation and classifiers
* `lists/` - Contains list of languages and dataset names used for trianing - these files are need for recipes in the steps/
* `etc/` - json files topic ID to int mapping
* `pylibs/` - add to PYTHONPATH
* `data_splits/` - Contains file IDs / doc IDs / rowIDs for creating 5 splits for MLDoc and INA

* To create MLDoc5x
  Use the give file IDs with the code from original MLDoc
* To create INA5x
  Dowload the original from IndicNLPsuite and run `src/generate_documents_indic_news_articles.py`

* See `steps/`
  0 through 6 for downloading data, preparing, extracting bow features, training mBay and cross-lingual classification

### Recipe

```bash
bash env.sh
```

* Download Europarl, MultiUN, NewsCommentary, GlobalVoices, for languages en, de, es, fr, it, ru.

```bash
steps/0_download_data_and_extract.sh lists/dataset_7L.list lists/langs_6L.list data/
```

```bash
steps/1_apply_msl_constraint_and_extract.sh lists/dataset_7L.list lists/langs_6L.list data/ 35
``````

```python
steps/2_create_flists_per_language.py data/ lists/dataset_7L.list -lang_list_file lists/langs_6L.list -msl 35 -out_flist_dir flists/eu-multiun-nc-gv/
```

* Create bag-of-words statistics using data for each language, independently. Here, we are limiting the max vocab size to `10000`, but it can be even higher.

```python
python steps/3_build_vocabulary_and_extract_bow_stats.py flists/eu-multiun-nc-gv/ exp/bow/eu-multiun-nc-gv/ -lang_list_file lists/langs_6L.list -xtr_flist_dir flists/eu-multiun-nc-gv/ -xtr_tag eu-multiun-nc-gv -mv 10000
```

```bash
steps/4_create_train_json.sh lists/dataset_7L.list lists/langs_6L.list exp/bow/eu-multiun-nc-gv/
```

```bash
steps/5_train_model_and_xtr_embeddings.sh exp/bow/eu-multiun-nc-gv/ 6L
```

### Citation
```
@misc{kesiraju2020bayesian,
      title={A Bayesian Multilingual Document Model for Zero-shot Topic Identification and Discovery},
      author={Santosh Kesiraju and Sangeet Sagar and Ondřej Glembek and Lukáš Burget and Ján Černocký and Suryakanth V Gangashetty},
      year={2020},
      eprint={2007.01359v3},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```