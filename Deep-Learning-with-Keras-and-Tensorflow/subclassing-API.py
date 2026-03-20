import tensorflow as tf

# Define your model by subclassing the Model class

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define layers in the constructor
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        # Define the forward pass in the call
        x = self.dense1(inputs)
        return self.dense2(x)

# Instantiate the model
model = MyModel()
# Define loss function and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
