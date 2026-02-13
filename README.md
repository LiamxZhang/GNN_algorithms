An implementation for torch-based GNN algorithms for task planning tasks.


## Prerequisites

    Operating System: Ubuntu 22.04
    NVIDIA Driver: Version 535.288.01 or higher
    CUDA Version: 12.1
    Anaconda/Miniconda: Installed on your system

## Installation

### Step 1: Clone this Project
Download the project code
```python
git clone https://github.com/LiamxZhang/GNN_algorithms.git
cd GNN_algorithms
```

### Step 2: Create Conda Environment
Create a new conda environment with Python 3.10
```python
conda create -n gnn_env python=3.10
conda activate gnn_env
```

### Step 3: Install dependencies

If the device is CPU only, install the pytorch in CPU version:
```python
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 cpuonly -c pytorch
```

If the device is Nvidia GPU, install the pytorch in CUDA version
```python
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install the torch-geometric library
```python
pip install --no-cache-dir torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
pip install --no-cache-dir torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
pip install --no-cache-dir torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
pip install --no-cache-dir torch-spline-conv==1.2.2 -f https://data.pyg.org/whl/torch-2.5.0+cpu.html

pip install torch-geometric==2.7.0
```

Install all rest dependencies through requirements.txt file

```python
pip install -r requirements.txt
```

## Tutorial
A tutorial of Jupyter notebooks can be found in the /tutorial folder. The content are
a_simple_tutorial_for_gcn: GCN, dataset, visualization
a_simple_tutorial_for_gat:

## Training