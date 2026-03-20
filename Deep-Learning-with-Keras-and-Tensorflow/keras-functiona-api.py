# Connect layers in a graph of layers. 
# The functional API is more flexible than the sequential API, 
# which is limited to a linear stack of layers. 
# The functional API allows you to create complex models, 
# such as multi-input/output models, directed acyclic graphs, 
# and models with shared layers.

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Define the input layer
inputs = Input(shape=(784,))
# Define the dense layer
x = Dense(64, activation="relu")(inputs)

# Define the output layer
outputs = Dense(10, activation="softmax")(x)


# Create the model
model = Model(inputs=inputs, outputs=outputs)


