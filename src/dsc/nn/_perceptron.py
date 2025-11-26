from __future__ import annotations
import numpy as np
import numpy.typing as npt


NDArrayFloat = npt.NDArray[np.float64]


class Peceptron:
    """
    Perceptron classifier.

    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0).
    n_iter : int
        Number of passes over the training set.
    random_state : int
        Seed for random number generator for random weight initialization.
    
    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.
    b_ : Scalar
        Bias unit after fitting.
    errors_ : list
        Number of misclassifications (updates) in each epoch.
    """

    def __init__(self, eta: float = 0.01, n_iter: int = 50, random_state: int = 1) -> None:
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

        self.w_: NDArrayFloat | None = None
        self.b_: np.float64 | None = None
        self.errors_: list[int] | None = None

    def fit(self, X: NDArrayFloat, y: NDArrayFloat):
        """
        Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of examples and 
            n_features is the number of features.
        y : array-like, shape = [n_examples]
            Target values.

        Returns
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.0)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

        return self

    def net_input(self, X: NDArrayFloat):
        """
        Calculate net input
        """
        assert self.w_ is not None
        assert self.b_ is not None
        return np.dot(X, self.w_) + self.b_

    def predict(self, X: NDArrayFloat):
        """
        Return class label after unit step.
        """
        return np.where(self.net_input(X) >= 0.0, 1, 0)


