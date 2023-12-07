
# Bayesian Subspace Multinomial Model (BaySMM)

* Model for learning document embeddings (i-vectors) along with their uncertainties.
* Gaussian linear classifier exploiting the uncertainties in document embeddings.
* See paper <http://arxiv.org/abs/1908.07599>

## Requirements

* Python >= 3.7
* PyTorch >= 1.1
* scipy >= 1.3
* numpy >= 1.16.4
* scikit-learn >= 0.21.2
* h5py >= 2.9.0

* See [INSTALL.md](INSTALL.md) for detailed instructions.

## Data preparation - sample from 20Newsgroups

* We will sample a subset with documents from 3 categories: `alt.atheism`, `sci.space`, `rec.autos`.

```python src/create_sample_data.py.py sample_data/```

## Training the model

* For help:

    ```python src/run_baysmm.py --help```

* To train on GPU set `CUDA_VISIBLE_DEVICES=$GPU_ID` where the `$GPU_ID` is the free GPU index

* Following code trains the model for `1000` VB iterations and saves the model in
an automatically created sub-directory: `exp/s_1.00_rp_1_lw_1e+01_l1_1e-03_50_adam/`

    ```python
    python src/run_baysmm.py train \
        sample_data/train.mtx \
        sample_data/vocab \
        exp/ \
        -K 50 \
        -trn 1000 \
        -lw 1e+01 \
        -var_p 1e+01 \
        -lt 1e-03
    ```

* ELBO and KLD for every iteration, log file, etc are saved in the sub-directory.

## Extracting the posterior distributions of embeddings

* Extract embeddings `[mean, log.std.dev]` for `1000` iterations for each of the stats file present in `sample_data/mtx.flist` file list.
* Using `-nth 100` argument,  embeddings for every `100`th iteration are also saved.

    ```python
    python src/run_baysmm.py extract \
        sample_data/mtx.flist \
        exp/s_1.00_rp_1_lw_1e+01_l1_1e-03_50_adam/model_T1000.h5 \
        -xtr 1000 \
        -nth 100
    ```

* Extracted embedding posterior distributions are saved in `exp/*/ivecs/` sub-directory with appropriate names.

## Training and testing the classifier

* Three classifiers can be trained on these embeddings.
* Use `--final` option to train and test classifier on embeddings from the final iteration.

1. Gaussian linear classifier - uses only the mean parameter

    ```python src/train_and_clf_cv.py exp/s_1.00_rp_1_lw_1e+01_l1_1e-03_50_adam/ivecs/train_model_T1000_e1000.h5 sample_data/train.labels glc```

2. Multi-class logistic regression - uses only the mean parameter

    ```python src/train_and_clf_cv.py exp/s_1.00_rp_1_lw_1e+01_l1_1e-03_50_adam/ivecs/train_model_T1000_e1000.h5 sample_data/train.labels lr```

3. Gaussian linear classifier with uncertainty - uses full posterior distribution

    ```python src/train_and_clf_cv.py exp/s_1.00_rp_1_lw_1e+01_l1_1e-03_50_adam/ivecs/train_model_T1000_e1000.h5 sample_data/train.labels glcu```

* All the results and predicted classes are saved in `exp/*/results/`

## Topic discovery

* Using the trained model and extracted embeddings, you can discover the topics in the dataset.
Each topic is be represented by a set of words.
* The topic discovery relies on k-means clustering.

* Below, we will cluster the embeddings into `k=20` clusters. Then consider only `topn=8` dense clusters
and display `topk=10` most representative words per cluster.

```python
python src/discover_topics.py \
        sample_data/vocab.json \
        exp/s_1.00_rp_1_lw_1e+01_l1_1e-03_50_adam/config.json \
        -ivecs_h5 exp/s_1.00_rp_1_lw_1e+01_l1_1e-03_50_adam/ivecs/test_model_T1000_e1000.h5
        -k 20 \
        -topn 8 \
        -topk 10
```

* Sample output is below (note that the original topics for the same data are: `alt.atheism`, `sci.space`, `rec.autos`)

```
Cluster   8: geology, retrieve, sfsuvax1, arthurc, chandler, sfsu, contacts, francisco, starflight, arthur,

Cluster  13: libemc, insurance, brigham, byuvm, geico, refund, dx, pointers, dealer, farm,

Cluster   6: bil, conner, osrhe, okcforum, theist, atheist, deliberately, theistic, validate, trustworthy,

Cluster   3: clutch, damaged, tires, miles, valve, prelude, tranny, milage, rod, chevy,

Cluster  18: oxides, gasses, depended, 626, 908, player, alex, delco, delcoelect, harvey,

Cluster   2: koresh, coutesy, follower, kaflowitz, campollo, wwc, 734928689, trilemma, gut, maddi,

Cluster   1: 60s, westminster, barlas, jkjec, shazad, toys, headlights, mliggett, crx, pontiac,

Cluster  10: servicing, grapple, usingthe, edo, costar, tug, hst, stow, unstow, gyros, ```
