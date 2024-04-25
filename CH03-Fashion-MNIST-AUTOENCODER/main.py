import numpy as np
from tensorflow.keras import datasets, utils, callbacks
from tensorflow.keras import layers, models
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

IMAGE_SIZE = 32
CHANNELS = 1
BATCH_SIZE = 100
BUFFER_SIZE = 1000
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 2
EPOCHS = 5

# Retrieve Fashion-MNIST data set
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

# Tune image data
def preprocess(imgs):
    # 0~255 to 0~1
    imgs = imgs.astype('float32') / 255.0
    # 28x28 to 32x32
    imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)
    imgs = np.expand_dims(imgs, -1)
    return imgs
x_train = preprocess(x_train)
x_test = preprocess(x_test)

# Encoder
encoder_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), name='encoder_input')
x = layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding='same')(encoder_input)
x = layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding='same')(x)
x = layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding='same')(x)
x_shape = K.int_shape(x) # (None, 4, 4, 128) None for batch_size we will know later when training.
shape_before_flattening = x_shape[1:] # we need this for decoder.
print('shape_before_flattening:',shape_before_flattening)
x = layers.Flatten()(x)
encoder_output = layers.Dense(EMBEDDING_DIM, name='encoder_output')(x)
encoder = models.Model(encoder_input, encoder_output)
encoder.summary()

# Decoder
decoder_input = layers.Input(shape=(EMBEDDING_DIM,), name='decoder_input')
# np.prod(shape_before_flattening) == 4 x 4 x 128
x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
x = layers.Reshape(shape_before_flattening)(x)
x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)
x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
decoder_output = layers.Conv2D(CHANNELS, (3, 3), strides=1, activation='sigmoid', padding='same', name='decoder_output')(x)
decoder = models.Model(decoder_input, decoder_output)
decoder.summary()

# Autoencoder
autoencoder = models.Model(encoder_input, decoder(encoder_output))
autoencoder.summary()

# Loss & Optimizer
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Create a model save checkpoint
model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath="./checkpoint",
    save_weights_only=False,
    save_freq="epoch",
    monitor="loss",
    mode="min",
    save_best_only=True,
    verbose=0,
)
tensorboard_callback = callbacks.TensorBoard(log_dir="./logs")

# Fit
autoencoder.fit(
    x_train,
    x_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(x_test, x_test),
    callbacks=[model_checkpoint_callback, tensorboard_callback]
)

# Save the final models
autoencoder.save("./models/autoencoder")
encoder.save("./models/encoder")
decoder.save("./models/decoder")

# Reconstructing
n_to_predict = 5000
example_images = x_test[:n_to_predict]
example_labels = y_test[:n_to_predict]
predictions = autoencoder.predict(example_images)

n_to_show = 10
indices = np.random.choice(range(len(example_images)), n_to_show)

fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, idx in enumerate(indices):
    pred_img = predictions[idx]
    ax = fig.add_subplot(1, n_to_show, i + 1)
    ax.axis("off")
    ax.imshow(pred_img)

    example_img = example_images[idx]
    ax = fig.add_subplot(2, n_to_show, i + 1)
    ax.axis("off")
    ax.imshow(example_img)
plt.show()

# Visualizing Latent Space
# 1. Show the encoded points in 2D space
embeddings = encoder.predict(example_images)
print(embeddings[:10])
plt.figure(figsize=(8,8))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c='black', alpha=0.5, s=3)
plt.show()

# 2. Colour the embeddings by their label (clothing type)
example_labels = y_test[:n_to_predict]
plt.figure(figsize=(8, 8))
plt.scatter(embeddings[:, 0], embeddings[:, 1], cmap='rainbow', c=example_labels, alpha=0.8, s=3)
plt.colorbar()
plt.show()


# Generating New Image
# Get the range of the existing embeddings
mins, maxs = np.min(embeddings, axis=0), np.max(embeddings, axis=0)
grid_width, grid_height = (6, 3)
sample = np.random.uniform(mins, maxs, size=(grid_width*grid_height, 2))
reconstructions = decoder.predict(sample)

# Render sample points in Latent Space
plt.figure(figsize=(8, 8))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c="black", alpha=0.5, s=2)
plt.scatter(sample[:, 0], sample[:, 1], c="#00B0F0", alpha=1, s=40)
plt.show()

# Render generated images
fig = plt.figure(figsize=(8, grid_height * 2))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(grid_width * grid_height):
    ax = fig.add_subplot(grid_height, grid_width, i + 1)
    ax.axis("off")
    ax.text(0.5, -0.35, str(np.round(sample[i, :], 1)), fontsize=10, ha="center", transform=ax.transAxes,)
    ax.imshow(reconstructions[i, :, :], cmap="Greys")
plt.show()

# Colour the embeddings by their label (clothing type - see table)
plt.figure(figsize=(12, 12))
plt.scatter(
    embeddings[:, 0],
    embeddings[:, 1],
    cmap="rainbow",
    c=example_labels,
    alpha=0.8,
    s=300,
)
plt.colorbar()

# Pick evenly distributed samples 15 x 15
grid_size = 15
x = np.linspace(min(embeddings[:, 0]), max(embeddings[:, 0]), grid_size)
y = np.linspace(max(embeddings[:, 1]), min(embeddings[:, 1]), grid_size)
xv, yv = np.meshgrid(x, y)
xv = xv.flatten()
yv = yv.flatten()
grid = np.array(list(zip(xv, yv)))

reconstructions = decoder.predict(grid)
fig = plt.figure(figsize=(12, 12))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(grid_size**2):
    ax = fig.add_subplot(grid_size, grid_size, i + 1)
    ax.axis("off")
    ax.imshow(reconstructions[i, :, :], cmap="Greys")
plt.show()