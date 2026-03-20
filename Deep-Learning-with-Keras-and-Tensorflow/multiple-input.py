
# Handling multiple inputs and outputs is also easy with the functional API.

from tensorflow.keras.layers import concatenate, Dense, Input

# Define two sets of inputs
inputA = Input(shape=(64,))
inputB = Input(shape=(128,))

# The first branch operates on the first input
x = Dense(8, activation="relu")(inputA)
x = Dense(4, activation="relu")(x)
x = Model(inputs=inputA, outputs=x)
# The second branch operates on the second input
y = Dense(16, activation="relu")(inputB)
y = Dense(4, activation="relu")(y)
y = Model(inputs=inputB, outputs=y)

# Combine the output of the two branches
combined = concatenate([x.output, y.output])
# Apply FC layer and then regression prediction on the combined outputs
z = Dense(2, activation="relu")(combined)
z = Dense(1, activation="linear")(z)

# The model will accept the inputs of the two branches and then output a single value
model = Model(inputs=[x.input, y.input], outputs=z)

