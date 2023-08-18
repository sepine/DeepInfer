# DeepInfer: Deep Type Inference from Smart Contract Bytecode


###  The artifact can be found at <https://github.com/sepine/DeepInfer>


# 1. Environment Installation
### Make sure the Anaconda has been installed in your computer. All the experiments are conducted on a Linux Server equipped with NVIDIA GeForce RTX 3090 24GB. Make sure your GPU environment has enough memory. <u>CPU is not supported.</u> <u>CUDA120 and above is not supported.</u>

## (1) Run the following command
```
conda env create -f deepinfer.yml
```
```
conda activate deepinfer
```

## Activate this environment and install the following libraries.

### (1) torch==1.11.0+cu113  
```
pip install torch-1.11.0+cu113-cp39-cp39-linux_x86_64.whl Please download the whl file from <https://download.pytorch.org/whl/torch/>
```
### (2) torch-scatter==2.0.9 and torch-sparse==0.6.15
```
pip install torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl 
pip install torch_sparse-0.6.15-cp39-cp39-linux_x86_64.whl
Please download the whl file from <https://pytorch-geometric.com/whl/torch-1.11.0%2Bcu113.html>  
```

### (3) torch-geometric==2.1.0.post1
```
pip install torch-geometric==2.1.0.post1
```

## Note: Please update the CUDA version according to your actual version for pytorch.

# 2. Dataset 
## Please download the dataset from <https://connectpolyu-my.sharepoint.com/:u:/g/personal/22037545r_connect_polyu_hk/Ecd6xOM9ttBBu6VF-vdYnhUBqhyobvvRKQ9CFghm_V6xzw?e=WHc8NY>

## <u>Note: the size of the file is about 90GB. the size of the unzipped file is about 300GB. Please make sure your PC have enought disk space.</u>

## Unzip the dataset and put them in the same directory as the main files as follows:
```
├─cached
    ├─compiler0.5
        ├─param
        ├─return
    ├─compiler0.6
        ├─param
        ├─return
    ├─compiler0.7
        ├─param
        ├─return
    ├─compiler0.8
        ├─param
        ├─return
    ├─solidity
        ├─param
        ├─return
    ├─vyper
        ├─param
        ├─return
├─datasets
    ├─compiler0.5
        ├─param
        ├─return
    ├─compiler0.6
        ├─param
        ├─return
    ├─compiler0.7
        ├─param
        ├─return
    ├─compiler0.8
        ├─param
        ├─return
    ├─solidity
        ├─param
        ├─return
    ├─vyper
        ├─param
        ├─return
├─models
    ├─compiler0.5
        ├─param
        ├─return
    ├─compiler0.6
        ├─param
        ├─return
    ├─compiler0.7
        ├─param
        ├─return
    ├─compiler0.8
        ├─param
        ├─return
    ├─solidity
        ├─param
        ├─return
    ├─vyper
        ├─param
        ├─return
├─xxx.py
```


# 3. Run the models 
## <u>Note: Please modify the number (i.e., batch size) according to your available GPU memory</u>
## (1) Parameter Prediction for Solidity
```
bash run_solidity_param.sh 128
```
## (2) Return Prediction for Solidity
```
bash run_solidity_return.sh 128
```
## (3) Parameter Prediction for Vyper
```
bash run_vyper_param.sh 128
```
## (4) Return Prediction for Vyper
```
bash run_vyper_return.sh 128
```
## (5) Parameter Prediction for Different Compiler Versions
```
bash run_compiler0.5_param.sh 128
```
```
bash run_compiler0.6_param.sh 128
```
```
bash run_compiler0.7_param.sh 128
```
```
bash run_compiler0.8_param.sh 128
```
## (6) Return Prediction for Different Compiler Versions
```
bash run_compiler0.5_return.sh 128
```
```
bash run_compiler0.6_return.sh 128
```
```
bash run_compiler0.7_return.sh 128
```
```
bash run_compiler0.8_return.sh 128
```
