## Basic Macine Learning and  Deep Learning Models

This repository consists of some of the basic models used in ML/DL. It is constantly updated and the current list of model are listed below:

Models Included:  __Linear Regression__ | __Linear Classification__ | __Logistic Regression__ | __Artificial Neural Networks (MNIST)__ | __Convolution Neural Netorks (CIFAR10)__ | __Convolution Neural Netorks (FashionMNIST)__| __Autoregressive Model__

>[!TIP]
> i) The trained model are saved in the models folders. For model structure used refere the code. <br/> ii) The data folder might consists some data but its mainly used as an place-hodler for your own data

### Setup the environment

1. [Install UV (Package Manager)](https://docs.astral.sh/uv/getting-started/installation/)


2. Git colne the repo to your working directory:

```sh

git clone https://github.com/rohitDey23/deeplearning_models.git

```
3. Get into the cloned repository:
```sh
cd deeplearning_models
```
4. Initalize UV to recognize the project as uv project
```sh
uv init
```
5. Download all the required dependincies based on _pyproject.toml_ file:
```sh
uv sync
```
If everything went well you wont see any error. To test if everything is working well run the following command.

```sh
uv run .\src\test_run.py 
```
This would print the version of python, numpy, pytorch. Also it should show if cuda device is available.

>[!CAUTION]
> Note: Its expected to have a NVIDIA GPU for the project, manual adjustemnt in _pyproject.toml_ would be needed for installing cpu version of putorch. 











