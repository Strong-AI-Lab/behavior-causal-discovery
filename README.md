# Behavior Caual Discovery

This project provides tools to build behavioral causal models for communities of social animals. Follow the instructions to build and evaluate the models. The data used in this project comes from the Meerkat Behaviour Recognition Dataset by [[Rogers et al., 2023]](https://arxiv.org/abs/2306.11326).

## Installation

Download the project:
```
git clone https://github.com/Strong-AI-Lab/behavior-causal-discovery.git
```

Instal the dependencies inside a virtual environment (or don't):
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Run

### Causal Structure Discovery

Run the following script to learn a causal model. This method recovers only the structure of the causal graph. See the next section to learn the transition probabilities.
```
python run_discovery.py
```

You can load an already learned model to re-generate the associated plots.
```
python run_discovery.py --save saves/<name-of-the-save-folder>
```

You can specify filter options to forbid unwanted dependencies (prediction for neighbors, low weight, uncovered causal direction). Use `--help` to get more details.
```
python run_discovery.py --filter neighbor_effect,low,corr
``` 

### Causal Inference

Run the following script to train a model. The model can either be a baseline model or a neural-causal inference model from a `run_discovery.py` save.
```
python train_model.py --model_type <model-type>
``` 

Provide the save folder to load the causal graph from if training a causal model.
```
python train_model.py --model_type <model-type> --save saves/<name-of-the-save-folder>
``` 


### Evaluation

Load the model for inference and evaluate it.
```
python run_inference.py saves/<name-of-the-save-folder>
```

Specify the model type to correctly load the weights of a parametric model.
```
python run_inference.py lightning_logs/version_n/<name-of-the-save-checkpoint>.ckpt --model_type <model-type>
```

### Train discriminator

Train a LSTM to distinguish the simulation generated by the model from the real distribution. Again, provide the save folder where the causal model is stored.
```
python run_discriminator.py saves/<name-of-the-save-folder>
```

You can load an already trained discriminator with the following option:
```
python run_discriminator.py saves/<name-of-the-save-folder> --discriminator_save saves/<name-of-the-discriminator-save-folder>
```