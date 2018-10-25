import tensorflow as tf

def build_mlp(input_placeholder, output_size, scope, n_layers, size, activation=tf.tanh, output_activation=None):
    """
        Builds a feedforward neural network
        
        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            output_size: size of the output layer
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of the hidden layer
            activation: activation of the hidden layers
            output_activation: activation of the ouput layers

        returns:
            output placeholder of the network (the result of a forward pass) 

        Hint: use tf.layers.dense    
    """
    output_placeholder = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            output_placeholder = tf.layers.dense(output_placeholder, size, activation=activation)
        output_placeholder = tf.layers.dense(output_placeholder, output_size, activation=output_activation)
    return output_placeholder