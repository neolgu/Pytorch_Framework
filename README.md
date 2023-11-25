# Pytorch_Framework

It is a framework for easy use of Pytorch. You can simply proceed with training and testing by adding the model and data loader as files. The project is basically implemented based on image classification.


## Framework Sturcture
```
|--checkpoint
|    |--model0
|        |--0.tar
|        |--1.tar
|        |--...
|    |--model1
|    |--model2
|--config
|    |--config.yaml
|--data
|    |--dataset.py
|    |--transform.py
|--model
|    |--model1.py
|    |--model2.py
|    |--...
|--train_test
|    |--train.py
|    |--test.py
|    |--utils.py
|--main.py
```

- checkpoint

  This is the location where the model’s training file is saved.
- config

  Various parameters to be used when performing train and test are specified through the yaml file.
- data

  This is the folder where you write code to load data to be used for learning. (Data loader)
- model

  This is the directory where model files to be used are collected.
- train_test

  Code for train and test
- main.py

  This is the executable file for the framework.

## Getting Started
### Get Clone
```
git clone https://github.com/neolgu/Pytorch_Framework
```

### Requirements

Basically it uses pytorch and PyYAML.

#### How to install [Pytorch](https://pytorch.org/get-started/locally/).

#### PyYAML
```
pip install PyYAML
conda install PyYAML
```

#### Other
```
pip install "if you want"
```

## Config.yaml
```yaml
# data parameters
dataset_name: dataset
dataset_path: dataset location ex) E:\Dataset\archive\trainingSet
save_path: mode save location ex) checkpoint/model0
resume: True of False (retrain parameter)
model_name: basemodel (same as model in model.py)
num_classes: 10 (in img classification)
epoch: 2
batch_size: 8
coder: N9
# training parameters
cuda: True
gpu_ids: [0]  # set the GPU ids to use, e.g. [0] or [1, 2]
num_workers: 16
lr: 0.0002
beta1: 0.9
beta2: 0.999
e: 1e-08
print_iter: 1000
# test parameters
load_path: checkpoint/model1/100.tar # load saved model
```

## Import your Model and Dataloader

First, copy the py file you want to use to the appropriate location.

```
|--data
|    |--dataset.py
|    |--transform.py
|--model
|    |--model1.py
|    |--model2.py
|    |--...
```

## Run main.py

if you want train.
```python
if __name__ == '__main__':
    trainer = Trainer("config/config.yaml")
    trainer.train()
```
if you want test.
```python
if __name__ == '__main__':
    tester = Tester("config/config.yaml")
    tester.test()
```

## Use it however you like

You can easily use deep learning models for other tasks by modifying some of the code.

In addition to the dataloader and model described above, you can apply loss functions, optimizers, etc. implemented in pytorch or directly.
