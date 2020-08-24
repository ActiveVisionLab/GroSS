# GroSS
This is the official code repository for the ECCV 2020 paper: ["GroSS: Group-Size Series Decomposition for Grouped Architecture Search"](https://gross.active.vision)

Currently, the released code supports decomposition, training and evaluation of the configurations included in the results section of the paper. We aim to release scripts to perform search soon.

# Installation
* Create and activate a new conda environment: `conda create -n gross python=3.7 & conda activate gross`
* Install pytorch with the correct version of CUDA for your machine (tested on PyTorch v1.3.1 and CUDA 10.1), eg: `conda install pytorch=1.3.1 torchvision=0.4.2 cudatoolkit=10.1 -c pytorch`
* Install the requirements: `pip install -r requirements.txt`

# Datasets
Various configurations use either CIFAR-10 or ImageNet.
For greatest convenience, download these datasets in a directory called `data`, such that the project structure is:
```
GroSS
├- configs
├- data
   ├- cifar
   └- imagenet
├- dataset
⋮
```
Alternatively, you can modify the DATASET.ROOT_DIR in the configuration files.

# Usage
To decompose and/or train a configuration run the command: `python decompose_and_ft.py -c PATH/TO/CHOSEN/CONFIGURATION/FILE`

For evaluation of a configuration, run the command: `python test_config.py -c PATH/TO/CHOSEN/CONFIGURATION/FILE`

# Citation
```
@misc{howardjenkins2019gross,
    title={GroSS: Group-Size Series Decomposition for Grouped Architecture Search},
    author={Henry Howard-Jenkins and Yiwen Li and Victor A. Prisacariu},
    year={2019},
    eprint={1912.00673},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```