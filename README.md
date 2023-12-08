# CS245 Final Project 

## GROUP: ST1

This project is an exploration of improvement for a dataset distillation algorithm: ["DREAM: Efficient Dataset Distillation by Representative Matching"](https://arxiv.org/abs/2302.14416).

## Requirements to replicating our results:
### Environment:
Be sure run code on Linux or Windows WSL2. Windows and macOS is not supported.

Make sure you have python version ```3.10``` and ```cuda version >= 12.0``` with Nvidia GPU supported.

Make sure you have a wandb account, we monitor our results on wandb.

Run```prepare.sh``` to install all packages needed for our project.

Or you can install packages using following command:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12 dask-cudf-cu12 cuml-cu12 cugraph-cu12 cuspatial-cu12 \
    cuproj-cu12 cuxfilter-cu12 cucim pylibraft-cu12 raft-dask-cu12


pip install matplotlib
pip install scikit-learn
pip install efficientnet_pytorch
pip install fast_pytorch_kmeans
pip install scikit-learn-extra
pip install scikit-learn-intelex
pip install wandb
```

### Replicating our results:
Run the following two Shell Script

```
run_1ipc.sh
```
```
run_10ipc.sh
```

Or you can use commands to run experiments step by step. For example, to run our experiment using DBSCAN clustering 
method with weight and ipc = 1, the command will be:
```angular2html
python condense.py --reproduce -d cifar10 -f 2 --ipc 1 --cluster_method DBSCAN --weight True
```

