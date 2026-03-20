# Define number of epochs
from xml.parsers.expat import model


epochs = 5

# Create a dummy training dataset
(train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(-1, 28 * 28).astype('float32') / 255

# Flatten and normalize the images
train_labels = train_labels.astype('int32')
# create a tf.data dataset for batching
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32)

# Custom training loop
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = model(x_batch_train, training=True)
            loss = loss_fn(y_batch_train, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            print(f"Epoch {epoch + 1}, Step {step + 1}, Loss: {loss.numpy():.4f}")
            