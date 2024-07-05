# Package imports
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
import argparse
import utils
import wandb

#define arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_size', type=int, default=300)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--reg', type=float, default=0.01)
parser.add_argument('--activation', type=str, default='relu', help='relu or tanh')
parser.add_argument('--hdim', type=int, default=2, help='number of nodes in hidden layer')
parser.add_argument('--print_loss', action='store_false', help='do not print loss')
parser.add_argument('--wandb', action='store_true')
args=parser.parse_args()
# Display plots inline and change default figure size
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

#generate dataset
np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)

#init wandb
if args.wandb:
    run = wandb.init(
        # Set the project where this run will be logged
        project="ML NLP",
        name=f'{args.activation}_reg:{args.reg}_lr:{args.lr}_hdim:{args.hdim}'
    )
    wandb.login()



num_examples = len(X) # training set size
nn_input_dim = X.shape[1] # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality


model, dataset, labels = utils.build_model(args, dataset=X, labels=y)



# Plot the decision boundary
utils.plot_decision_boundary(lambda x: utils.predict(model, x, args), dataset, labels)
plt.title(f"Decision Boundary for hidden layer size {args.hdim}"),

