# Experiment Plan

## Learning Question

Can a small fully connected neural network learn a useful Fashion-MNIST classifier using a clean PyTorch training loop, fixed seed, validation split, and saved experiment artifacts?

## Dataset

- Dataset: Fashion-MNIST.
- Source path: `torchvision.datasets.FashionMNIST`.
- Data root: `/scratch/rui.gao/ai_research_roadmap_data`.
- Registry entry: `data_registry/datasets.yaml`.
- Preparation status: user reported successful preparation.

## Split Plan

Use the official Fashion-MNIST train/test split.

Use the official training split for model selection:

- training: 50,000 examples
- validation: 10,000 examples

Use the official 10,000-example test split only for final evaluation after choosing the baseline or comparison run.

## Baseline

Initial baseline proposal:

- model: one-hidden-layer MLP
- input: flattened 28 x 28 image
- hidden size: 128
- activation: ReLU
- loss: cross-entropy
- optimizer: Adam
- learning rate: 0.001
- batch size: 64
- epochs: 5
- seed: fixed before dataset split and model initialization

## Comparison Runs

Planned comparison runs:

1. Baseline MLP.
2. Learning-rate variant.
3. Hidden-size variant.

Do not add more runs until these three are understood and documented.

## Metrics

Track at minimum:

- training loss
- validation loss
- validation accuracy
- final test accuracy

Optional later metrics:

- per-class accuracy
- confusion matrix
- runtime
- CPU/GPU memory notes

## Falsification Condition

This setup is not useful enough as a Week 2 baseline if:

- the training loop cannot finish a short run
- validation accuracy does not improve beyond a weak baseline
- results are not reproducible from the saved settings
- metrics and artifacts are not saved clearly enough to compare runs

## Expected Artifacts

Each meaningful run should produce:

- command used
- settings used
- metrics summary
- notes on failures or surprises
- loss and accuracy plot
- checkpoint, if implementation is authorized later

Large checkpoints must stay out of Git.

## Open Decisions

- Exact learning-rate variant.
- Exact hidden-size variant.
- Whether the first run should happen locally, on the HPC login environment, or through SLURM.
- Whether to add an offline synthetic smoke-test path when implementation is authorized.


## Fashion-MNIST dataset

- The Fashion-MNIST dataset have images that are 28 * 28 grayscale
- Eache immage has 1 channel (it's grayscale, so no rgb)
- flattened input size is 28 * 28 = 784 
- There are 10 clothing classes. 
- The model predicts one of the 10 labels. 

## Splitting the data

- 50,000 and 10,000 split. 10,000 good enough for split
- split should be random. without replacement.

## The baseline Model:

MLP, simple neural network. NO cnns.

Input is 784 samples and then 128 nodes, to 10 numbers (then with these 10 numbers we do softmax)

Loss is cross-entropy since we are using softmax. 

# Models:

Baseline Model: 
input 784, hidden layer: 128, activation: ReLU. output: 10 logits,
loss: CrossEntropyLoss, optimizer: Adam, learning rate:0.001, batch size:64, epochs: 5

learning rate variant:
hidden size: 128
learning rate: 0.003

hidden size variant:
hidden size: 256
learning rate: 0.001


# Files

dataset.py
model.py
train.py
eval.py 
tests/test_smoke.py