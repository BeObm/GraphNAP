# Graph Neural Architecture Prediction

GraphNAP is an Automatic Graph Neural Architecture framework which provides an efficient GNN automatic modeling approach based on the neural predictor.

The framework of GraphNAP is as follows:


####  Highly customizable

  - GraphNAP supports user-defined almost all module functions easily.


## Installing For Ubuntu 16.04

- **Ensure you have installed CUDA 10.2 before installing other packages**

**1. Nvidia and CUDA 10.2:**

```python
[Nvidia Driver] 
https://www.nvidia.cn/Download/index.aspx?lang=cn

[CUDA 10.2 Download and Install Command] 
#Download:
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
#Install:
sudo sh cuda_10.2.89_440.33.01_linux.run

```

**2. Python environment:** recommending using Conda package manager to install

```python
conda create -n graphnap python=3.7
source activate graphnap
```

**3. Pytorch V1.8.1:** execute the following command in your conda env graphnap

```python
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch
```

**4. Pytorch Geometric:** execute the following command in your conda env graphnap
```python
conda install pyg -c pyg
```

**5. Matplotlib:** execute the following command in your conda env graphnap
```python
conda install -c conda-forge matplotlib
```

## User-defined
GraphNAP is very friendly for users to implement customization, users can freely define their own functional components as long as they follow the **custom specification** and GraphNAP will automatically load user-defined components. Users can know the **custom specification** of each functional component in the following list, which is very simple. The list of definable components is as follows: 

**1. Search Space**

To custom the search space, please follow this steps:

1- Go to [GraphNAP/search_space_manager/search_space.py](search_space_manager/search_space.py) and add or remove a components or options form the search space. Note that the search space is defined as a dictionary where key are the search space component names and values are a list of corresponding  options 

2- Go to [GraphNAP/search_space_manager/map_function](search_space_manager/map_functions.py) to map each new options to its corresponding class if applicable

3- If there is a new component is added, please go to [GraphNAP/search_algo/utils.py](search_algo/utils.py) and update the function _get_edge_index_  accordingly 

**Remark**: The current version of the code is specifically built  for _2-layer_ GNN. We recommend users to not change this at this moment. We are working on the update to make it flexible 

 
**2. Configuration parameters**

All configuration parameters can be changed in the [configuration file](settings/config_file.py) , which is easy to understand

* **Type task** : The type of task can be modified at line 40. The type of class refers to the downstream task and can take any value in ["node classification", "graph classification"]

* **dataset source** : The source of the dataset can be modified at ligne 41. The dataset source list is from [Pytorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/data_cheatsheet.html) and includes "_TUDataset_" for graph classification and "_Planetoid_" for node classification

*  **dataset name** : The name of the dataset can be modified at line 42. Note that the name of the dataset should be the same as appearing in pytorch-geometric for the given dataset source

*  **new dataset**: To add a new dataset, please refer to [load_data.py](load_data/load_data.py)

*  **Search global parameters**: Search parameters can be modified from line 51 to 76

*  **predictor model**: predictor training parameters, can be modified from line 89 to 9

* **Paths**: Files management configurations can be set in _create_paths_ function 


## Default Configuration

For default configuration, we refer users to [GrapNAP/settings/config+file.py](settings/config_file.py)

## Datasets
 
All datasets used in this paper are publicly available and can be downloaded using  [PyTorch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html).  

## How to run GraphNAP ?
 To run GraphNAP, execute the following command in your conda env graphnap 
```python
python main.py
```

## Where to get Outputs ?

 Outputs of GraphNAP will be saved in result files as defined in _[create_paths_ function ](settings/config_file.py)
 

_**Remark**_: We keep modifying our code. Updated version will be uploaded once available.