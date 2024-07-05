To run the extensive hyperparameter search run run.sh

nn_from_scratch.py is the main script that accepts the following arguments:

dataset_size: int
lr: float (learning rate)
reg:float (regularization strength)
activation: "relu" or "tanh"
hdim: int (number of hidden layer dimensions)
print_loss: store_false (prints loss every 6000 steps)
wandb: store_true (enable wand logging)