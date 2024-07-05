import numpy as np
import matplotlib.pyplot as plt
import wandb
def plot_decision_boundary(pred_func, dataset, labels):
    # Set min and max values and give it some padding
    x_min, x_max = dataset[:, 0].min() - .5, dataset[:, 0].max() + .5
    y_min, y_max = dataset[:, 1].min() - .5, dataset[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(dataset[:, 0], dataset[:, 1], c=labels, cmap=plt.cm.Spectral)

def calculate_loss(model, dataset, labels, args):
    num_examples=len(dataset)
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = dataset.dot(W1) + b1
    if args.activation == 'relu':
        a1=np.maximum(z1,0)
    else:
        a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), labels])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += args.reg/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    if args.wandb:
        wandb.log({ "loss": 1./num_examples * data_loss})
    return 1./num_examples * data_loss

# Helper function to predict an output (0 or 1)
def predict(model, x, args):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    if args.activation =='relu':
        a1=np.maximum(z1,0)
    else:
        a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(args, dataset, labels, num_passes=20000):
    num_examples=len(dataset)
    nn_input_dim=dataset.shape[1]
    nn_output_dim=2
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, args.hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, args.hdim))
    W2 = np.random.randn(args.hdim, nn_output_dim) / np.sqrt(args.hdim)
    b2 = np.zeros((1, nn_output_dim))

    # This is what we return at the end
    model = {}
    loss_list=[]
    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        z1 = dataset.dot(W1) + b1
        if args.activation=='relu':
            a1=np.maximum(z1,0)
          
        else:
            a1=np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), labels] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        if args.activation=='relu':
            # delta2 = (delta3.dot(W2.T) >= 0).astype(int)
            # print(a1)
            delta2 = delta3.dot(W2.T) * ((a1 > 0).astype(int))
            # print(delta2)
        else:
            delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
            # print(delta2)
        dW1 = np.dot(dataset.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += args.reg * W2
        dW1 += args.reg * W1

        # Gradient descent parameter update
        W1 += -args.lr * dW1
        b1 += -args.lr * db1
        W2 += -args.lr * dW2
        b2 += -args.lr * db2
        
        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        
        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        loss = calculate_loss(model, dataset,labels, args)
        loss_list.append(loss)
        if args.print_loss and i % 6000 == 0 :
        #   print("Loss after iteration %i: %f" %(i, calculate_loss(model, dataset, labels,  args)))
            print("Loss after iteration %i: %f" %(i, loss))
    loss_array=np.array(loss_list)
    np.save(f'./results/{args.activation}_reg:{args.reg}_lr:{args.lr}_hdim:{args.hdim}.npy', loss_array)
    return model, dataset, labels