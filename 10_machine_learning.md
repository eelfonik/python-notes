## Machine learning

The `numpy` lib

Assume we have imported the numpy lib as `np`

- `np.arange(10)` will create an array with values from 0 to 9
- `np.random.randn(10000)` will create a 10000 x 10000 matrix with random values
- `np.dot(A,B)` will multiply the matrix A and matrix B => _Note_ that `np.dot()` performs a **matrix-matrix** or **matrix-vector** multiplication. This is different from `np.multiply()` and the `*` operator (which is equivalent to .* in Matlab/Octave), which performs an **element-wise multiplication**.

	ex: `np.dot(x, x)` 可以用来算x的平方。无论x是一个raw number, 还是一个matrix
	
- `np.zeros((n,m))` will create a n*m matrix with only zeros.
- `sum` & `reshape` methond on matrix:

	```python
	# a 3 x 4 matrix
	X = np.array([[56.0, 0.0, 4.4, 68.0],
              [1.2, 104.0, 52.0, 8.0],
              [1.8, 135.0, 99.0, 0.9]
             ])
	# axis=0 means sum up vetically
	# use axis=1 to sum up horizontally
	cal = X.sum(axis=0)

	# here as the cal is already a 1 x 4 matrix, we don't have to use reshape
	# but it does no harm you call reshape to make sure it's in the right shape
	percentage = 100*X/cal.reshape(1,4)
	
	# we can also use `shape` to get the dimension of a matrix or vector
	print(percentage)
	```
	
	- A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b ∗ c ∗ d, a) is to use:
	
	```python
	X_flatten = X.reshape(X.shape[0], -1).T 
	# X.T is the transpose of X
	```
	
	
**NOTE:** Also make sure to check the numpy **broadcasting** to see how it apply _automatic transform_ on values/matrix to perform calculations like `*`, `/` and `+/-`

#### some tips on using numpy vectors

- always specify the shape of matrix when using `random.randn()`:

	for example, if you use `A = random.randn(5)` to initiate a matrix, it will be a shape of `(5,)`, so if you then use `A.T` to **transpose** the matrix, you'll get the same shape `(5,)`, which is not you might want. _Side note_: this kind of matrix is actually called **rank 1 array**, avoid using it!
	
	So use `A = random.randn(5,1)` to make sure it's a 5 x 1 matrix, with a shape of `(5,1)`, then when use `A.T`, you'll get a matrix of shape `(1,5)`
	
	You can also make sure the matrix (or we say vector) is in right shape by adding assertion like `assert(a.shape == (5, 1))`, then you'll get an `AssertionError` if it's not right.
	
	If for any reason that your data is a rank 1 array, you can always use `reshape` to make it the right shape.
	
	And always remember to add assertions.
	
-
	
#### Simple example of binary classification using sigmoid function & a convex function as Loss function (which means it has only one local optima to make things simpler)

![img](https://www.dropbox.com/s/6n3zkuk2qqhrbjt/Screen%20Shot%202017-08-24%20at%2012.50.12%20AM.png?raw=1)

First we need to import data:

- a helper loader with name `lr_utils`:

```python
import numpy as np
import h5py
    
    
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
```

- import data:

```python
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
# this helper loader is defined somewhere else, might using h5py
from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
```

Then Process(clean up) data:

```python
# Figure out the dimensions and shapes of the problem
# (m_train, m_test, num_px, ...)

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

# Reshape the training and test examples

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

# center and standardize dataset
# meaning that you substract the mean of the whole numpy array from each example,
# and then divide each example by the "standard deviation" of the whole numpy array

# But for picture datasets,
# it is simpler and more convenient and works almost as well
# to just divide every row of the dataset by 255
# (the maximum value of a pixel channel).

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
```

Start coding: 

- some different activition Fns

Activition Fns (**non-linear Fns**) are needed because if you use a direct output of `z=w*x+b`, the deep neuro network makes no sense, as no matter how many **layers** you used, it will always be linear to the initial inputs, that makes **multiple layers** in network useless.

The only place you might want to use a linear fn (aka `g(z) = z`), is on **output layer** where you want the output to be a _raw number_, like predicting house prices.

![img](https://www.dropbox.com/s/hp4jbrhefhigaid/Screen%20Shot%202017-08-24%20at%206.16.31%20PM.png?raw=1)


```python

# ---------different activition Fns------------

# 1. sigmoid
# often use only on "output layer" of binary classification
# as its value is between [0,1]

# deriative of z => dz = A(1-A), with A as a matrix of result 
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    s = 1/(1+np.exp(-z))
    
    return s

# 2. hyperbolic tangent function
# has better effect when apply on hidden layers
# since the value is centered at 0, in a range(-1,1)
# instead of 0.5 as sigmoid fn
# tanh is part of the numpy library, you don't have to define it yourself

# deriative of z => dz = 1-A*A, with A as a matrix of result 
def tanh(z):
	"""
    Compute the hyperbolic tangent of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- tanh(z)
    """
    
    s = (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z)) 
    
    return s
    
# 3. reLu function
# often used on hidden layers (most common)

# deriative of z => dz = 1 when z >=0, dz = 0 when z < 0
def relu(z):
	"""
    Compute the relu of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- relu(z)
    """
    
    s = np.maximum(0,z) 
    
    return s
    
# 4. leaky relu function

# deriative of z => dz = 1 when z >=0, dz = 0.01 when z < 0
def leaky_relu(z):
	"""
    Compute the leaky relu of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- leaky relu(z)
    """
    
    s = np.maximum(0.01*z, z) 
    
    return s

    
# --------------end of different activition Fns--------------
```

- an example with _sigmoid_ fn as activition Fn, includes **init**, **propagate**, **optimize (update)** and **predict**:


```python
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
   
    w = np.zeros((dim, 1))
    b = 0
    
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b
    
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    
    # compute activation, produce a matrix the same shape as Y
    # w.T = (1, n), X = (n,m)
    # so np.dot(w.T,X) will be (1,n)dot(n,m) results (1,m)
    
    # in fact np.dot(w.T,X) can be seen as
    # 	temp1 = w * X (即 (n,1) * (n,m) => broadcasting => (n,m))
    #  sum = temp1.sum(axis=0) => vertically相加，变为（m,)
    # 	sum.reshape(1,temp1.sum(axis=0).shape[0]) => reshape成(1,m)
    
    A = sigmoid(np.dot(w.T, X) + b)
    
    # compute cost
    # `Y*np.log(A)+(1-Y)*np.log(1-A)` will be the same shape as Y = (1,m)
    # sum this vector up horizontally use `np.sum` will endup with shape (1,1)
    
	cost = -np.sum(Y*np.log(A)+(1-Y)*np.log(1-A), axis =1, keepdims = True)/m

    # BACKWARD PROPAGATION (TO FIND GRAD)
    
    # (A-Y).shape=(1,m), X.shape=(n,m)
    # so (n,m)dot(m,1) will result (n,1), which is the same shape as w
    
    # actually we can make a `dz` to clarify it's the partical of sigmoid fn
    # dJ => -(Y*np.log(A)+(1-Y)*np.log(1-A))求导 => dJ = -Y/A + (1-Y)/(1-A)
    # dz => 用dJ乘上sigmoid的导数(A(1-A))即可 => dz = dJ * A(1-A) = A-Y
    # 同理如果activition fn用的是tanh => dz = dJ * (1-A*A) = 
    
    # 所以针对不同的activaton Fn,我们可以得到不同的dz,再带入给dw,db
    dz = A-Y

    dw = np.dot(X, dz.T)/m
    db = np.sum(dz, axis = 1, keepdims = True)/m

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
    
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs
    
    
def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0][i] > 0.5:
            Y_prediction[0][i] = 1
        else:
            Y_prediction[0][i] = 0
        pass
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction
    
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    
    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    # optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False)
    parameters, grads, costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples
    # predict(w, b, X)
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
    
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

```

##### Note:

if you increase the number of iterations on training set, you might see that the training set accuracy goes up, but the test set accuracy goes down. This is called **overfitting**. =>就是说这个模型太靠近于用来training的data, 对于测试data不太友好。

##### About Learning rate:

The learning rate α determines how rapidly we update the parameters. If the learning rate is too large we may "overshoot" the optimal value. Similarly, if it is too small we will need too many iterations to converge to the best values. That's why it is crucial to use a *well-tuned learning rate*.

##### About intialization random rather than zeros:

For a network with hidden layers with multiple nodes, if you init all the `W` and `B` to be zero, it means **all the units (or nodes)** in the _same hidden layer_ will compute exactly the same way for both forward & backward propagation, that makes **multiple nodes** in one layer useless.

So use `W = np.random.randn((n,m)) * 0.01`, but it's ok to init `B = np.zeros((n,1))`. 

**N.B.** 这里`W`需要乘上`0.01`是因为我们不希望W的值太大，否则经过activition Fn（特别是sigmoid和tanh这类的Fn）计算出的A的值会过大或过小，导致直接落在非常平缓的两端曲线上，会影响gradient decent的计算效率。如果是一个deep network，我们可能会用其他的constant,而不是`0.01`， 但都不会太大。

-

#### Logistic regression did not work well on the "flower dataset".

Which means the dataset is not linearly separable. In that case, we can try using a network with hidden layers.

- A network model with one hidden layer:

![img](https://www.dropbox.com/s/pmgjq3oh5dc8e50/Screen%20Shot%202017-08-25%20at%204.26.40%20PM.png?raw=1)

```python
# assume we have imported all necessary libs already, and loaded our data

X, Y = load_planar_dataset()

def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of nodes in the hidden layer (hard coded 4 here)
    n_y -- the size of the output layer
    """
    
    n_x = X.shape[0] # size of input layer
    n_h = 4
    n_y = Y.shape[0] # size of output layer
    return (n_x, n_h, n_y) 

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters
    """
    
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))
    
    # if we have multiple hidden layers (i)
    # we chould add n_h_1, n_h_2,...n_h_i, etc, then
    # Wi = np.random.randn(n_h_i, n_h_i-1) * 0.01
    # bi = np.zeros((n_h_i, 1))
    
    # EX:
    # layer_dims = [2, 4, 5, 6, 7]
    # for l in range(1, len(layer_dims)):
    #   parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
    #   parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters 
    
def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    ''' parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}'''
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (Y.shape[0], X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache
    
def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, m)
    Y -- "true" labels vector of shape (1, m)
    parameters -- python dictionary containing parameters W1, b1, W2 and b2
    
    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1] # number of example

    # Compute the cross-entropy cost
    logprobs = np.multiply(Y, np.log(A2)) + np.multiply(1-Y, np.log(1-A2))
    cost = -np.sum(logprobs)/m
    
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
    assert(isinstance(cost, float))
    
    return cost

def backward_propagation(parameters, cache, X, Y):
    """
    Arguments:
    parameters -- python dictionary containing parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, m)
    Y -- "true" labels vector of shape (1, m)
    
    Returns:
    grads -- python dictionary containing gradients with respect to different parameters
    """
    m = X.shape[1]
    
    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters["W1"]
    W2 = parameters["W2"]
        
    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache['A1']
    A2 = cache['A2']
    
    # Backward propagation: calculate dW1, db1, dW2, db2. 
    dZ2= A2-Y
    # dZ2 is calculated as the (loss Fn)' * (sigmoid fn)'
    # where sigmoid is the activition Fn of the output layer
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis=1, keepdims = True)/m
    
    # dZ1 is dA1 * (tanh fn)', where dA1 = W2.T * dZ2
    dZ1 = np.dot(W2.T, dZ2) * (1-np.power(A1,2))
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis=1, keepdims = True)/m
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads
    
def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent
    
    Arguments:
    parameters -- python dictionary containing parameters 
    grads -- python dictionary containing gradients 
    
    Returns:
    parameters -- python dictionary containing updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    
    # Update rule for each parameter
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, m)
    Y -- labels of shape (1, m)
    n_h -- size of nodes in the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    # Initialize parameters, then retrieve W1, b1, W2, b2.    
    parameters = initialize_parameters(n_x, n_h, n_y)
#     W1 = parameters['W1']
#     b1 = parameters['b1']
#     W2 = parameters['W2']
#     b2 = parameters['b2']
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):
         
        # Forward propagation
        A2, cache = forward_propagation(X, parameters)
        
        # Cost function
        cost = compute_cost(A2, Y, parameters)
 
        # Backpropagation.
        grads = backward_propagation(parameters, cache, X, Y)
 
        # Gradient descent parameter update.
        parameters = update_parameters(parameters, grads)
        
        
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters  
    
def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(X, parameters)
    predictions = A2 > 0.5
    
    return predictions
    
# ---------Put them all togother--------------

# Build a model with a n_h-dimensional hidden layer
parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)

# Try predict accuracy
predictions = predict(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')

```

##### Some takeaways:

- to set the entries of a matrix X to 0 and 1 based on a threshold you would do: `X_new = (X > threshold)`
- Additionally, we can try different hidden layer units value, according to the result, we can apply **regularization** to use very large models (such as n_h = 50) without much overfitting.
- We have also something called **Early stopping**, which is a way to prevent overfitting

```python
plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations = 5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
    
'''
Accuracy for 1 hidden units: 67.5 %
Accuracy for 2 hidden units: 67.25 %
Accuracy for 3 hidden units: 90.75 %
Accuracy for 4 hidden units: 90.5 %
Accuracy for 5 hidden units: 91.25 %
Accuracy for 20 hidden units: 90.0 %
Accuracy for 50 hidden units: 90.25 %
'''
```

- _Check dimensions of matrix:_
	- `W(i) = (n(i), n(i-1))`
	- `b(i) = (n(i), 1)`
	- `Z(i) = (n(i), m)`

	Where `i` is referring to the `ith` layer in the network, and `n(i)` referring to the number of nodes in `ith` layer. So given an input `X`, and a true result `Y`:
	
	- for the 1st layer, `W1.shape == (n1, X.shape[0])`
	- for the last output layer, `Wlast.shape == (Y.shape[0], n(last-1))`
	- **N.B.** here we can see, as `Alast.shape == (Y.shape[0], X.shape[1])`, we must have `Y.shape[1] == X.shape[1] == m`
	
	And `m` is the number of training examples.
	
-
#### Benefits of deeper NN:

![img](https://www.dropbox.com/s/siie1lfy35k9kya/Screen%20Shot%202017-08-25%20at%207.40.25%20PM.png?raw=1)

With `log(n)` layers of network, we can reduce the units needed in a single hidden layer, which will be `exp(n)`

For a complete code example for L layers NN, see [here](https://github.com/eelfonik/python-notes/blob/master/deep_learning.py).


	