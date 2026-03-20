from tensorflow.keras.layers import concatenate, Dense, Input

# Define two sets of inputs
inputA = Input(shape=(32,))
inputB = Input(shape=(32,))


# The first branch operates on the first input
x = Dense(8, activation="relu")(inputA)
x = Dense(4, activation="relu")(x)

# The second branch opreates on the second input
y = Dense(16, activation="relu")(inputB)
y = Dense(4, activation="relu")(y)


# Combine the output of the two branches
combined = concatenate([x, y])


# Apply a dense layer and then a regression prediction on the combined outputs
z = Dense(2, activation="relu")(combined)
z = Dense(1, activation="linear")(z)

# The model will accept the inputs of the two branches and then output a single value
model = Model(inputs=[inputA, inputB], outputs=z)


# Compile the model
model.compile(optimizer='adam', 
              loss='mean_squared_error', 
              metrics=['accuracy'])