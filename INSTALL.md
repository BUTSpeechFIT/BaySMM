### Required packages
```
torch numpy scipy scikit-learn numexpr h5py tqdm  matplotlib sentencepiece jieba konoha
```

### Detailed steps for installation

* Create a `python3` virtual environment

```bash
mkdir -p envs
python3 -m venv envs/mbay-env
```

* Activate the environment

```bash
. envs/mbay-env/bin/activate
```

* Install required packages

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

```bash
pip3 install scikit-learn scipy numexpr tqdm h5py matplotlib
```

```bash
pip3 install sentencepiece jieba konoha
```

* Create `path.sh` to activate the environment and set PYTHONPATH

```bash
echo ". ${PWD}/envs/mbay-env/bin/activate" > path.sh
echo "export PYTHONPATH=${PWD}/mbay/:${PYTHONPATH}" >> path.sh
echo "export PYTHONPATH=${PWD}/pylibs/:${PYTHONPATH}" >> path.sh
chmod u+x path.sh
```

* Run `. path.sh` and you are ready.



