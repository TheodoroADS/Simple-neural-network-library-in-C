# Simple neural network library in C from scratch

This project consists in simple functions for creating multi layer perceptrons. I have created this project to better understand how a neural network works, and tried to implement its core algorithms without looking things up too much on the internet (especially backpropagation), so it is not very well optimised... I have tested my functions on the MNIST dataset, and got up to ~= 94% precision on the test set, the code for this example is in main.c.


## ***How to use the functions for creating a MLP*** 

## Creating a network

### 1. Create the NN struct

Use the NN_create function. This function takes parameters of the network you wish to create and returns a pointer to an instance of the NN struct representing it.

``` C 

    //function for creating a network
    NN* NN_create(int input_size, 
    int batch_size, 
    int output_size,
    Output_Activation_func output_activation, 
    Loss_func loss_function);

```

**Parameters of NN_create**

Here are the parameters that NN_create takes in:

* input_size : The size of the input layer of the network 

* batch_size : The batch size for SGD

* output_size : The size of the output layer

* output_activation : A function pointer to the activation function of the output layer. A valid output activation function is defined in activation.h as:

    ``` C 

    typedef void (*Output_Activation_func)(size_t , double*); 

    ```

    For now, the sigmoid activation function and softmax are implemented in activation.c

* loss_function : A pointer for the loss function used by the network. A valid loss function is defined in loss.h as:

    ``` C 
    typedef double (*Loss_func)(size_t, double*, double*);
    ```

    For now, Categorical Cross Entropy and Mean Square Error are available and implemented in loss.c


### 2. Add hidden layers

Once your network is created, you can add hidden layers with the NN_add_hidden function, which takes the NN struct pointer and the parameters of the hidden layer you wish to add:

``` C
    int NN_add_hidden(NN* network ,int layer_size, void (*layer_activation)(double*));

```

Other than the pointer to the network, this function accepts layer_size, which is the number of neurons of the layer, and layer_activation, the pointer to the activation function of this layer. 

A valid layer activation function shape is defined in activation.h as:

``` C
    typedef void (*Activation_func)(double*); 
```

For now, sigmoid, relu and leaky_relu are available and implemented in activation.c.

### 3. Compile your network

When you are done adding layers, you must call the function NN_compile to finish preparating your network for training. This function will, for example, allocate the weights for the output layer, which is something that can only be done once all the hidden layers are added. 

``` C 
    int NN_compile(NN* network);
```

## Training a network

For now, only a function appropriate for training a network in classification tasks is implemented. The core difference of this existing function to a another one fit for regression is that it accepts the labels of the training examples as integers and converts them to one-hot inside the body of the function. A function fit for regression is coming soon!

For classification tasks you can use NN_fit_classification: 

```C
    void NN_fit_classification(NN* network,
    size_t nb_examples,
    size_t nb_epochs,
    double** values,
    int* labels,
    double learning_rate);
```

This function will apply fit the network to the data provided (I tried to imitate the Keras library's fit function) and run the SGD algorithm over a fixed number of epochs. 

To use it you must provide the following arguments: 

* network : a pointer to the network you want to train

* nb_examples : the number of examples provided in the training set

* nb_epochs : the number of epochs you want to train the network over

* values : a pointer to an array containing the data to be passed as the training set. This is a 2D array: the first dimension contains each individual example. Each example is itself an array containing the values of the input for this example (this values are of the double type). For example, for the MNIST dataset, each example is an array containing the values for every pixel of the image, and every example is contained in an array. This big array is the "values" input parameter. 

* labels : an array containing the label for each example as an integer. For the MNIST example, a label would be the digit the image represents as an integer value.

* learning_rate : the learning rate hyper-parameter for the backpropagation algorithm.


## ***Project organisation***

It consists in 4 main source files: 

* matrix.c : matrix logic and structure

* activation.c : where the different activation functions are defined

* loss.c : where the different loss functions are defined

* nn.c : where the main logic for the neural network is