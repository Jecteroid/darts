# DARTS 🎯 playground 🧗‍

### This repo is made for ML2's playground projects
> **🌏 running environment info** <br>
> `python >= 3.6, pytorch == 1.0, and needs CUDA`
> <br><br>
> **Requirements** <br>
> ```
  torch
  torchvision
  graphviz
  numpy
  tensorboard
  tensorboardx```

<br>

### 🚀 How to search and train?
> 🎲 Simply, you can run DARTS search process with <br> &nbsp;&nbsp;&nbsp;&nbsp; `python run.py --name <your_pjt_name> --dataset <data_NAME> --data_path <your_PATH>` <br><br>
> --> ex) `python run.py --name DARTS_test1 --dataset cifar10 --data_path ../data`
> 
> If you need customize some parameters, check `python run.py -h`
>
> This process can visualize by using tensorboard <br>
> (after run.py execute)`tensorboard --logdir=./searchs/<your_pjt_name>/tb --port=6006`<br>
>
> you can visualize with python visualize.py DARTS
<br>

### 🔗 Process description. 🥚🐣🐥
#### 1. start setting
> 1. Get some arguments in shell
> 2. Set training environment such as using GPU
> 3. Define model(Network) and optimizers
> 4. Make Dataset(dataloader) -- cifar10
> 5. Set lr scheduler
> 6. and Define arch 

#### 2. under training (arch searching)
> 1. start epoch loop
> 2. ├ set lr scheduler 
> 3. ├ set genotype
> 4. ├ start training
> 5. ⎪ ├ start step loop (batch streaming)
> 6. ⎪ ⎪ ├ dataset setting
> 7. ⎪ ⎪ ├ arch stepping (architecture weight)
> 8. ⎪ ⎪ ⎪ ├ 
> 8. ⎪ ⎪ ⎪ ├ backward
> 9. ⎪ ⎪ ⎪ ├ optimizer step
> 6. ⎪ ⎪ ├ model training
> 10.⎪ ⎪ ├ model fitting()
> 11. and making now...

#### 3. under training (arch searching)



This project is referred from

- DARTS https://arxiv.org/abs/1806.09055

- git https://github.com/quark0/darts
