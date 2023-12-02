# Must Run at Linux or WSL2 with Nvidia GPU support (CUDA Version >= 12)
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


