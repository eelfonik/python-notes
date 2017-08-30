## Hyperparameters

Things like:

-  `learning_rate`
-  number of `iterations`
-  number of `hidden layers`
-  number of `hidden units` in each layer.
-  `Activition Fns` for hidden layers
-  _I think_ also the `Loss Fn` for determine the loss(performance) of the model


-
**The things of setting up :**

### how to divide the dataset

- training set (Tr) => used to training models
- cross validation / development set (D) => try to find which model works best
- testing set (Te) => testing on the final choosen model

If you've got small dataset, normally it'll be good to keep the ratio at `Tr/D/Te = 60/20/20`.

But for big data, like for 1,000,000 or more data entries, we chould make the training set much bigger, like `Tr/D/Te = 98/1/1`, or even `99.5/0.4/0.1`.

### mismatch on train/test distribution

As models need a lot training data to train, it's possible that we use images crawling from the web as *training set*, and our *dev/test set* are from the pictures uploaded by users, which might have mismatch.

The rule of thumb could be to always make sure that _**dev and test set** are coming from the same distribution_.

### Bias/Variance

![img](https://www.dropbox.com/s/m7r0mkjgyjaglit/Screen%20Shot%202017-08-26%20at%203.05.07%20AM.png?raw=1)

- if errors in training set is low, but in dev set is much higher, then we have an _overfitting_, thus a **high variance**. 
- if both errors rate are _similar_, but are both much higher than human level or **Optimal (Bayers) error**, then we have an _underfitting_, thus a **high bias**.


##### some methods suggested:
- for high bias, which means the train set not perform well, we could:

	- bigger network
	- training longer
	- (NN architecture search)

- for high variance, which means the dev set not perform well, we could: 

	- get more data
	- regularization
	- (NN architecture search)


##### Some **regularization** techniques to deal with high variance (_overfitting_)

Regularization的所有目的就是防止`W`过大。

- **_L2 (Frobenius Fn) regularization_** => 在计算cost的时候加入一个L2 Fn, with a parmameter `lambd`, 以便penalize the `W` from being too large. 

![img](https://www.dropbox.com/s/qe4aud5v9p8d6pf/Screen%20Shot%202017-08-30%20at%2012.49.59%20AM.png?raw=1)

```python
# Compute cost assume a 3 layers network like this: 
# LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.

m = X.shape[1]

cross_entropy_cost = compute_cost(A3, Y)
L2_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) / (2 * m)

cost = cross_entropy_cost + L2_cost


# backward prop: =>
# add the regularization term's gradient for dWl.
dWl = 1./m * np.dot(dZl, A(l-1).T) + lambd/m * Wl

```

- **_Dropout regularization_** => 随机扔掉每一层的几个nodes,把network变小。**But don't drop out at test time**


Ex: _Inverted dropout_ for a certain layer `l = 3`

```python
keep-prob = 0.8

# Forward: =>
#create a boolean maxtrix the same shape as a3

# here note difference between `np.random.randn` and `np.random.rand`
# the later will randomly get numbers between 0 and 1
D3 = np.random.rand(A3.shape[0], A3.shape[1]) < keep-prob 
A3 = A3 * D3
# this line is to ensure that the `a3` as an input to next layer
# has expected value
A3 = A3/keep-prob

# backward: =>

dZ4 = np.multiply(dA4, np.int64(A4 > 0))
# TODO: why does the above line needed???
dA3 = np.dot(W4.T, dZ4)
# apply the mask matrix again from cache (to shut down the same nodes)
dA3 = dA3 * D3
# During forward propagation, you had divided a3 by keep_prob.
# In backpropagation, you'll therefore have to divide da3 by keep_prob again
# to scale the value of neurons that haven't been shut down
dA3 = dA3 / keep_prob


```

**Note 1:** the `keep-prob` can be different to different layers, so make it smaller on bigger matrix, and even can be `1` to small martix like 1 by 1.

**Note 2:** if there's no overfitting problem, you don't need to use regularization at all, but in some certain fields (like computer vision), as the original inputs (`X`) tends to be really large (lots of pixels), it's almost a default to  use dropout. 

**Note 3:** *deactive* dropout(or make `keep-prob` to 1) at first: as the dropout will make the cost Fn `J` less well defined, it can not play nicely for the debug plot showing _J decreases when iteration number increased_. So deactive it first, after making sure that J will decrease with more iteration, turn it back on. => 又例如在计算backprop gradient的时候，先把dropout deactive.

- **_Data augmentation_**

Ex: if you don't have enough training data for images, you can flip horizontally the images, or scale/zoom the images to increase the size of training set.

- **_early stopping_**

![img](https://www.dropbox.com/s/ftwrio2xjf2m68v/Screen%20Shot%202017-08-27%20at%206.59.21%20PM.png?raw=1)

Plot also the **dev set errors**, then **stop** training your model at the number of iteration *where dev errors started to go up*.

**Note:** Downside of early stopping, is it tries to both *optimize* the cost Fn `J`, and *prevent* overfitting at the same time. So it won't do good at both.


### Optimization

#### Normalize inputs data

Why?

如果你的input data里的 x1, x2的range很不同的话（例如x1的范围是 0->1, 而x2的范围是1000->2000），那么`W`和`b`相对loss fn `J`的三维plot会变得非常扁平，并且不那么symmetric （即`w`和`b`太不对称了）, 导致在做iteration的时候，learning_rate必须很小，而且每次iteration不见得就往local optima笔直前进了。因此normalization可以让你的training过程更快一些。

1. subtract mean: 

	```python
	mean = np.sum(X, axis = 1)/m
	X = X - m
	```
2. Normalize variance

	```python
	sig = np.sum(X ** 2, axis =1)/m
	X = X/sig
	```


#### Vanishing/exploding gradient for very deep network

If the network is very deep, then the activation Fn will decrease/increase exponentially of `l`, so the final result can be extreme small or large, and make the calculation of gradient very slow.

A partial solution would be carefully choose your init `W`:

- for larger input size, make W smaller (assume we have a **relu** as activation Fn) => `wl = np.random.randn(n,n-1) * np.sqrt(2/n-1)`

The term `np.sqrt(2/n-1)` is called variance of the activation Fn

for **relu** Fn, the default variance would normally be `np.sqrt(2/n-1)` => **He Initialization**.

for **tanh** Fn, the default variance would normally be `np.sqrt(1/n-1)`, or `np.sqrt(2/(n + n-1)` => **Xavier Initialization**. 

The number `2` or `1` is also a hyperparam you can tune, but normally we won't touch that much if not have tuned other hyperparams.


#### Check gradient to make sure backprop right

Use it only when debug, don't use in training as it's slow.

- take `w1`,`b1`,...`wn`, `bn` and make a giant vector `theta`, so that the loss Fn `J` is `J(theta)`.
- take also `dw1`,`db1`, ... `dwn`, `dbn`, and make a `d(theta)`.
- for each i, calc `dapprox(theta)`, with a `epsilon = 10e-7`
- check `dapprox(theta)` with `d(theta)` using L2 fn

![img](https://www.dropbox.com/s/pmsm3le481v448k/Screen%20Shot%202017-08-30%20at%201.16.35%20AM.png?raw=1)

**Note:** this computation doesn't work with _dropout_, as it randomly eliminate nodes.

```python
def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    count = 0
    for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        
        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1,1))
        keys = keys + [key]*new_vector.shape[0]
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys

def vector_to_dictionary(theta):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}
    parameters["W1"] = theta[:20].reshape((5,4))
    parameters["b1"] = theta[20:25].reshape((5,1))
    parameters["W2"] = theta[25:40].reshape((3,5))
    parameters["b2"] = theta[40:43].reshape((3,1))
    parameters["W3"] = theta[43:46].reshape((1,3))
    parameters["b3"] = theta[46:47].reshape((1,1))

    return parameters

def gradients_to_vector(gradients):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """
    
    count = 0
    for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3"]:
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1,1))
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta
    
def gradient_check_n(parameters, gradients, X, Y, epsilon = 1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n
    
    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters. 
    x -- input datapoint, of shape (input size, 1)
    y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
    
    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    
    # Set-up variables
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    
    # Compute gradapprox
    for i in range(num_parameters):
        
        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        # "_" is used because the function you have to outputs two parameters but we only care about the first one
        thetaplus = np.copy(parameters_values)                                      
        thetaplus[i][0] = thetaplus[i][0] + epsilon                                
        J_plus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaplus))                                   
        
        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        thetaminus = np.copy(parameters_values)                                     
        thetaminus[i][0] = thetaminus[i][0] - epsilon                                      
        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaminus))                                  
        
        # Compute gradapprox[i]
        gradapprox[i] = (J_plus[i] - J_minus[i])/ (2 * epsilon)
    
    # Compare gradapprox to backward propagation gradients by computing difference.
    numerator = np.linalg.norm(grad - gradapprox)                                           
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)                                         
    difference = numerator / denominator                                          

    if difference > 1e-7:
        print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    
    return difference

```




