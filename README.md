# MSGLON_NS

This repository provides code to reproduce the experiment described in “Novelty-Based Generation of Continuous Landscapes with Diverse Local Optima Networks”.

## Step1. Require Environment
python: 3.11  
Operating System: Windows 10/11 64-bit  
GPU: NVIDIA GPU with CUDA 12.1+  (Those experiments can be run without using a GPU)  

## Step2. Install Python
Download installer: https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
```terminal
python --version
```

## Step3. Install NVIDIA Driver (If using GPU)
Download: https://www.nvidia.com/en-us/drivers/  
Install "Game Studio Driver"  
```terminal
nvidia-smi
```


## Step4. Install CUDA Toolkit 12.1.1
Download: https://developer.nvidia.com/cuda-12-1-1-download-archive

```terminal
nvcc --version
```


## Step5. Clone this repository

```terminal
git clone https://github.com/KPIMZT/Diverse_LON.git
```

## Step6. Create and Activate a Python Virtual Environment
```terminal
cd Diverse_LON
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python --version
```

```terminal
python -m pip install --upgrade pip
```

## Step7. Install Dependencies
Install pytorch with CUDA:
```terminal
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining requirements:
```terminal
pip install -r requirements.txt
```

## Step8. Download NS result data
If you want to replicate the experiments after the benchmark without running NS, please download the data from Zenodo (https://doi.org/10.5281/zenodo.19630354)

directory structure  

Diverse_LON/  
-results_NS/  
-results_benchmark/  
-results_cor/  


## Step9. Trace our experiments
RQ1
Compare BoA
```terminal
python BoA_ex.py
```

RQ2
Novelty Search:
```terminal
python NS_ex.py
```

RQ3
benchmark, correlation and reggression analysi:
```terminal
python benchmark_cor_reg_ex.py
```









