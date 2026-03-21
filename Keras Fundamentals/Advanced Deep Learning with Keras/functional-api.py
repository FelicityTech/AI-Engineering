from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Define the input layer
inputs = Input(shape=(784,))

# Define the hidden layer
x = Dense(64, activation='relu')(inputs)
# Define the output layer
outputs = Dense(10, activation='softmax')(x)

# Create the model
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
