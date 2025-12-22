import numpy as np

class Adeline:
    """
    ADAptive LInear NEuron classifier.

    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0).
    n_iter : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for random weight initialization.

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.
    b_ : Scalar
        Bias unit after fitting.
    losses_ : list
        Mean squared error loss function values in each epoch.
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1) -> None:
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of training 
            instances and n_features is the number of input features.
        y : array-like, shape = [n_examples]
            Target values.
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.0)
        self.losses_ = []

        print("Beginning training.")
        # For each pass through the training data...
        for i in range(self.n_iter):
            print("".join(["-"]*80))
            print(f"Epoch {i}")
            print("".join(["-"]*80))
            # Compute the net input for the network.
            net_input = self.net_input(X)
            # Pass the net input through the chosen activation function.
            output = self.activation(net_input)
            # Calculate the per instance errors. 
            errors = (y - output)
            # Update the weight vector (Partial derivative wrt. each weight).
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            # Update the bias term (Partial derivative wrt. the bias term).
            self.b_ += self.eta * 2.0 * errors.mean()
            # Calculate the loss for the current epoch. MSE in this example.
            loss = (errors**2).mean()
            print(f"Loss: {loss}")
            self.losses_.append(loss)
            print("".join(["-"]*80), end="\n\n")
        print("Training complete!")

        return self


    def net_input(self, X):
        """
        Calculate net input.

        Takes the sum of products of input x-values and corresponding weights, 
        then adds the bias vector.
        """
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """
        Compute linear activation.

        The activation function is the identity function in this example.
        """
        return X

    def predict(self, X):
        """
        Return class label after unit step.

        Unit step function here is:
            - z_i >= 0.0 -> 1
            - z_i < 0.0 -> 0

        The threshold function would only use a decision boundary at 0.5 if a 
        probablistic scale is used (ex. sigmoid or logistic activation 
        function). The activation function here is identity.
        """
        # Calculate the net input and pass it through activation function.
        raw_pred = self.activation(self.net_input(X))
        # Apply the threshold fucntion to get final prediction.
        return np.where(raw_pred >= 0.0, 1, 0)

