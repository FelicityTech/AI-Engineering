from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.models import Model

# First input Model
inputA = Input(shape=(32, 32, 1))
x = Conv2D(32, (3, 3), activation=relu)(inputA)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Model(inputs=inputA, outputs=x)

# Second input Model
inputB = Input(shape=(32, 32, 1))
y = Conv2D(32, (3, 3), activation=relu)(inputB)
y = MaxPooling2D((2, 2))(y)
y = Flatten()(y)
y = Model(inputs=inputB, outputs=y)

# Combine the output of the two branches
combined = concatenate([x.output, y.output])
# Apply a FC layer and then a regression prediction on the combined outputs
z = Dense(64, activation=relu)(combined)
z = Dense(1, activation=linear)(z)
# The model will accept the inputs of the two branches and then output a single value
model = Model(inputs=[x.input, y.input], outputs=z)

# Compile the model
model.compile(optimizer='adam', 
              loss='mean_squared_error', 
              metrics=['accuracy'])
