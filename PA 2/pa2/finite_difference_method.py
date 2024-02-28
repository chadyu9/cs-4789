import numpy as np


def gradient(f, x, delta=1e-5):
    """
    Returns the gradient of function f at the point x
    Parameters:
        f (numpy.array -> double): A scalar function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method


    Returns:
        ret (numpy.array): gradient of f at the point x
    """
    # TODO
    (n,) = x.shape

    # Each element of the gradient is just the partial derivative of f with respect to x_i
    return np.array(
        [
            (
                f(x + delta * np.eye(1, n, i).flatten())
                - f(x - delta * np.eye(1, n, i).flatten())
            )
            / (2 * delta)
            for i in range(n)
        ]
    )


def jacobian(f, x, delta=1e-5):
    """
    Returns the Jacobian of function f at the point x
    Parameters:
        f (numpy.array -> numpy.array): A function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method

    Returns:
        ret (numpy.array): A 2D numpy array with shape (f(x).shape[0], x.shape[0])
                            which is the jacobian of f at the point x
    """
    # TODO
    (n,) = x.shape
    (m,) = f(x).shape
    x = x.astype("float64")  # Need to ensure dtype=np.float64 and also copy input.
    # i^th row of Jacobian is just the gradient of f_i
    return np.array([gradient(lambda x: f(x)[i], x, delta) for i in range(m)])


def hessian(f, x, delta=1e-5):
    """
    Returns the Hessian of function f at the point x
    Parameters:
        f (numpy.array -> double): A scalar function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method

    Returns:
        ret (numpy.array): A 2D numpy array with shape (x.shape[0], x.shape[0])
                            which is the Hessian of f at the point x
    """
    # TODO
    # Hessian is just the Jacobian of gradient of f
    return jacobian(lambda x: gradient(f, x, delta), x, delta)
