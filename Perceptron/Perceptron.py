def activation(h):
    """Activation of a neuron, defined using threshold function."""

    if(h > 0):
        return 1

    else:
        return 0


def update_weight(wij, yj, tj, xi, lr = 0.25):
    """Update a weight Wij for a misclassified sample, based on a simple error formula and learning rate."""

    new_wij = wij - lr * ((yj - tj) * xi)
    new_wij = round(new_wij, 3)
    #print("\t", wij, "-", lr, "* (", yj, "-", tj, ") *", xi, "=", new_wij)

    return new_wij

def weighted_sum(W, X):
    """Calculate the weighted sum for a neuron, given its input and weight vectors."""

    if len(W) != len(X):
        print("Dimension of weight vector should be same as input vector.")
        return

    else:
        H = 0

        for i in range(len(W)):
            H += (W[i] * X[i])
    
        return H

def forward_pass(X, target_Y, W):
    """Complete a forward pass for one input sample through the perceptron."""

    pred_Y = activation(weighted_sum(W, X))
    print("\tI/P:", X, "   O/P:", target_Y, "   W:", W, "   W_Sum:", round(weighted_sum(W, X), 3))

    if pred_Y != target_Y:
        for j in range(len(W)):
            W[j] = update_weight(W[j], pred_Y, target_Y, X[j])

    return W

def perceptron_learning(train_data, W, epoch = 3):
    """Driver function to run the learning mechanism for the perceptron. """
    
    for T in range(epoch):
        print("\nEpoch:", T + 1)

        for i in range(len(train_data)):
            X = train_data[i][0]
            target_Y = train_data[i][1]
            W = forward_pass(X, target_Y, W)
            print("\tUpdated Weights: {0}\n".format(W))
        
        print("")

    return W


if __name__ == "__main__":
    print("\n\t\tSimple Perceptron Learning Algorithm\n")

    #For OR Gate
    
    train_data =    [[[-1, 0, 0], 0]
                    ,[[-1, 0, 1], 1]
                    ,[[-1, 1, 0], 1]
                    ,[[-1, 1, 1], 1]]

    W = [-0.05, -0.02, 0.02]

    W = perceptron_learning(train_data, W, epoch = 4)
            

"""
~/Desktop/Machine Learning 
➜ python Perceptron.py

		Simple Perceptron Learning Algorithm


Epoch: 1
	I/P: [-1, 0, 0]    O/P: 0    W: [-0.05, -0.02, 0.02]    W_Sum: 0.05
	Updated Weights: [0.2, -0.02, 0.02]

	I/P: [-1, 0, 1]    O/P: 1    W: [0.2, -0.02, 0.02]    W_Sum: -0.18
	Updated Weights: [-0.05, -0.02, 0.27]

	I/P: [-1, 1, 0]    O/P: 1    W: [-0.05, -0.02, 0.27]    W_Sum: 0.03
	Updated Weights: [-0.05, -0.02, 0.27]

	I/P: [-1, 1, 1]    O/P: 1    W: [-0.05, -0.02, 0.27]    W_Sum: 0.3
	Updated Weights: [-0.05, -0.02, 0.27]



Epoch: 2
	I/P: [-1, 0, 0]    O/P: 0    W: [-0.05, -0.02, 0.27]    W_Sum: 0.05
	Updated Weights: [0.2, -0.02, 0.27]

	I/P: [-1, 0, 1]    O/P: 1    W: [0.2, -0.02, 0.27]    W_Sum: 0.07
	Updated Weights: [0.2, -0.02, 0.27]

	I/P: [-1, 1, 0]    O/P: 1    W: [0.2, -0.02, 0.27]    W_Sum: -0.22
	Updated Weights: [-0.05, 0.23, 0.27]

	I/P: [-1, 1, 1]    O/P: 1    W: [-0.05, 0.23, 0.27]    W_Sum: 0.55
	Updated Weights: [-0.05, 0.23, 0.27]



Epoch: 3
	I/P: [-1, 0, 0]    O/P: 0    W: [-0.05, 0.23, 0.27]    W_Sum: 0.05
	Updated Weights: [0.2, 0.23, 0.27]

	I/P: [-1, 0, 1]    O/P: 1    W: [0.2, 0.23, 0.27]    W_Sum: 0.07
	Updated Weights: [0.2, 0.23, 0.27]

	I/P: [-1, 1, 0]    O/P: 1    W: [0.2, 0.23, 0.27]    W_Sum: 0.03
	Updated Weights: [0.2, 0.23, 0.27]

	I/P: [-1, 1, 1]    O/P: 1    W: [0.2, 0.23, 0.27]    W_Sum: 0.3
	Updated Weights: [0.2, 0.23, 0.27]



Epoch: 4
	I/P: [-1, 0, 0]    O/P: 0    W: [0.2, 0.23, 0.27]    W_Sum: -0.2
	Updated Weights: [0.2, 0.23, 0.27]

	I/P: [-1, 0, 1]    O/P: 1    W: [0.2, 0.23, 0.27]    W_Sum: 0.07
	Updated Weights: [0.2, 0.23, 0.27]

	I/P: [-1, 1, 0]    O/P: 1    W: [0.2, 0.23, 0.27]    W_Sum: 0.03
	Updated Weights: [0.2, 0.23, 0.27]

	I/P: [-1, 1, 1]    O/P: 1    W: [0.2, 0.23, 0.27]    W_Sum: 0.3
	Updated Weights: [0.2, 0.23, 0.27]


"""