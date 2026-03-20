# Shared layers apply the same layer instance to different inputs. 
# This is useful when you want to share weights between layers, 
# as in a Siamese network.
from tensorflow.keras.layers import Dense, Lambda

# Define a shared layer
input = Input(shape=(28, 28, 1))
conv_base = Dense(64, activation='relu')

# Process the input through the shared layer
processed_1 = conv_base(input)
processed_2 = conv_base(input)

# Create a model using the shared layer
model = Model(inputs=input, outputs=[processed_1, processed_2])