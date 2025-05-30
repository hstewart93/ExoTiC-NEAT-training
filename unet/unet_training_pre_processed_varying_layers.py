import os
from datetime import datetime

ROOT_DIR="/data/typhon2/hattie/jwst/soss_simulations/unet_training_data/processed_data_2000"
SAVE_DIR=f"save_{datetime.now().isoformat()}"
os.makedirs(os.path.join(ROOT_DIR, SAVE_DIR), exist_ok=True)

os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"

print(f"Root directory: {ROOT_DIR}")
print(f"Save directory: {SAVE_DIR}")

import numpy as np
import tensorflow as tf

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    MaxPooling2D,
    Dense,
)

from keras.models import Model
from keras.optimizers import Adam, schedules
from matplotlib import pyplot as plt

# Check for GPU
gpu_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpu_devices))

class Unet:
    """UNet model for image segmentation."""

    def __init__(
        self,
        input_shape: tuple,
        filters: int = 16,
        dropout: float = 0.05,
        batch_normalisation: bool = True,
        trained_model: str = None,
        image: np.ndarray = None,
        layers: int = 4,
        output_activation: str = "sigmoid",
        model: Model = None,
        reconstructed: np.ndarray = None,
        kernel_size: tuple = (3, 3),
        # _kernel_sizes: list[tuple] = [(5, 15), (5, 15), (3, 11), (3, 11), (3, 5), (3, 5)],
        # _kernel_sizes: list[tuple] = [(9, 71), (5, 39), (3, 23), (3, 11), (3, 5), (3, 3)],
        # _kernel_sizes: list[tuple] = [(9, 17), (5, 13), (5, 9), (5, 9), (5, 7), (3, 5)],
        _kernel_sizes: list[tuple] = [(9, 17), (5, 5), (5, 9), (5, 9), (5, 7), (3, 3)],
    ):
        """
        Initialise the UNet model.

        Parameters
        ----------
        input_shape : tuple
            The shape of the input image.
        filters : int
            The number of filters to use in the convolutional layers, default is 16.
        dropout : float
            The dropout rate, default is 0.05.
        batch_normalisation : bool
            Whether to use batch normalisation, default is True.
        trained_model : str
            The path to a trained model.
        image : np.ndarray
            The image to decode. Image must be 2D given as 4D numpy array, e.g. (1, 256, 256, 1).
            Image must be grayscale, e.g. not (1, 256, 256, 3). Image array row columns must
            be divisible by 2^layers, e.g. 256 % 2^4 == 0.
        layers : int
            The number of encoding and decoding layers, default is 4.
        output_activation : str
            The activation function for the output layer, either sigmoid or softmax.
            Default is sigmoid.
        model : keras.models.Model
            A pre-built model, populated by the build_model method.
        reconstructed : np.ndarray
            The reconstructed image, created by the decode_image method.
        """
        self.input_shape = input_shape
        self.filters = filters
        self.dropout = dropout
        self.batch_normalisation = batch_normalisation
        self.trained_model = trained_model
        self.image = image
        self._kernel_sizes = _kernel_sizes
        self.layers = len(_kernel_sizes)

        self.output_activation = output_activation
        self.model = model
        self.reconstructed = reconstructed
        self.kernel_size = kernel_size

        self.model = self.build_model()

    def convolutional_block(self, input_tensor, filters, kernel_size):
        """Convolutional block for UNet."""
        convolutional_layer = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            kernel_initializer="he_normal",
            padding="same",
        )
        batch_normalisation_layer = BatchNormalization()
        relu_layer = Activation("relu")

        if self.batch_normalisation:
            return relu_layer(batch_normalisation_layer(convolutional_layer(input_tensor)))
        return relu_layer(convolutional_layer(input_tensor))

    def encoding_block(self, input_tensor, filters, kernel_size):
        """Encoding block for UNet."""
        convolutional_block = self.convolutional_block(input_tensor, filters, kernel_size)
        max_pooling_layer = MaxPooling2D((2, 2), padding="same")
        dropout_layer = Dropout(self.dropout)

        return convolutional_block, dropout_layer(max_pooling_layer(convolutional_block))

    def decoding_block(self, input_tensor, concat_tensor, filters, kernel_size):
        """Decoding block for UNet."""
        transpose_convolutional_layer = Conv2DTranspose(
            filters, (kernel_size), strides=(2, 2), padding="same"
        )
        skip_connection = Concatenate()(
            [transpose_convolutional_layer(input_tensor), concat_tensor]
        )
        dropout_layer = Dropout(self.dropout)
        return self.convolutional_block(dropout_layer(skip_connection), filters, kernel_size)

    def build_model(self):
        """Build the UNet model."""
        input_image = Input(self.input_shape, name="img")
        current = input_image

        # Encoding Path
        convolutional_tensors = []
        for layer, ks in enumerate(self._kernel_sizes):
            convolutional_tensor, current = self.encoding_block(
                current, self.filters * (2 ** layer), ks
            )
            convolutional_tensors.append((convolutional_tensor))

        # Latent Convolutional Block
        latent_convolutional_tensor = self.convolutional_block(
            current, self.filters * 2 ** self.layers, self._kernel_sizes[-1],
        )

        # Decoding Path
        current = latent_convolutional_tensor
        for layer, ks in reversed([i for i in enumerate(self._kernel_sizes)]):
            current = self.decoding_block(
                current, convolutional_tensors[layer], self.filters * (2 ** layer), ks
            )

        # outputs = Conv2D(1, (1, 1), activation=self.output_activation)(current)
        # outputs = tf.abs(Dense(1, activation=self.output_activation)(current))
        outputs = tf.abs(Conv2D(1, (1, 1), activation=self.output_activation)(current))

        model = Model(inputs=[input_image], outputs=[outputs])
        return model

    def compile_model(self):
        """Compile the UNet model."""
        self.model.compile(
            optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy", "iou_score"]
        )
        return self.model

    def decode_image(self):
        """Returns images decoded by a trained model."""
        print(f"Predicting source segmentation using pre-trained model...")
        if self.trained_model is None or self.image is None:
            raise ValueError("Trained model and image arguments are required to decode image.")
        if isinstance(self.image, np.ndarray) is False:
            raise TypeError("Image must be a numpy array.")
        if len(self.image.shape) != 4:
            raise ValueError("Image must be 4D numpy array for example (1, 256, 256, 1).")
        if self.image.shape[3] != 1:
            raise ValueError("Input image must be grayscale.")
        if (
            self.image.shape[0] % 2 ** self.layers != 0
            and self.image.shape[1] % 2 ** self.layers != 0
        ):
            raise ValueError("Image shape should be divisible by 2^layers.")

        self.model = self.compile_model()
        self.model.load_weights(self.trained_model)
        self.reconstructed = self.model.predict(self.image)
        return self.reconstructed


data_train = np.load(os.path.join(ROOT_DIR, "data_train.npy"))
labels_train = np.load(os.path.join(ROOT_DIR, "labels_train.npy"))

data_validation = np.load(os.path.join(ROOT_DIR, "data_validation.npy"))
labels_validation = np.load(os.path.join(ROOT_DIR, "labels_validation.npy"))

print(f"Data train shape: {data_train.shape}")
print(f"Labels train shape: {labels_train.shape}")

print(f"Data validation shape: {data_validation.shape}")
print(f"Labels validation shape: {labels_validation.shape}")

training_dataset = tf.data.Dataset.from_tensor_slices((data_train, labels_train))
validation_dataset = tf.data.Dataset.from_tensor_slices((data_validation, labels_validation))

batch_size = 8
training_dataset = training_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

unet_model = Unet(input_shape=(256, 2048, 1), output_activation=None, layers=6, kernel_size=(5, 15))
model = unet_model.model

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mean_squared_error",
    metrics=["accuracy"],
)

print(model.summary())

class CustomCallBack(Callback):

    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs["loss"])
        self.val_losses.append(logs["val_loss"])

        self.logs_losses = np.array({"loss": self.losses, "val_loss": self.val_losses})
        
        np.save(os.path.join(ROOT_DIR, SAVE_DIR, "training_loss_varying_layers_2000_test_run_zeroth_scaled_1e7.npy"), self.logs_losses)

        fig, ax = plt.subplots(1, figsize=(5, 4))
        ax.plot(self.losses, label="training loss", color="#57d4c1", marker="None")
        ax.plot(self.val_losses, label="validation loss", color="#8d57d4", marker="None")

        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()

        plt.savefig(os.path.join(ROOT_DIR, SAVE_DIR, "training_loss_plot_varying_layers_2000_test_run_zeroth_scaled_1e7.png"))

        ax.set_yscale("log")
        plt.savefig(os.path.join(ROOT_DIR, SAVE_DIR, "training_loss_plot_varying_layers_log_2000_test_run_zeroth_scaled_1e7.png"))

        plt.close()

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint(
        os.path.join(ROOT_DIR, SAVE_DIR, "trained_model_varying_layers_2000_test_run_zeroth_scaled_1e7.weights.h5"),
        verbose=1,
        save_best_only=True,
        save_weights_only=True
    ),
    CustomCallBack(),
]


results = model.fit(
    training_dataset,
    # data_train,
    # labels_train,
    # batch_size=8,
    epochs=400,
    callbacks=callbacks,
    validation_data=(data_validation, labels_validation),
)

best_model_epoch, best_model_val_loss = np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"])

fig, ax = plt.subplots(1, figsize=(5, 4))

ax.plot(results.history["loss"], label="training loss", color="#57d4c1", marker="None")
ax.plot(results.history["val_loss"], label="validation loss", color="#8d57d4", marker="None")
ax.plot(best_model_epoch, best_model_val_loss, marker="x", label="best model", color="#d457bf", linestyle="None", markersize=5)

ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.legend()

plt.savefig(os.path.join(ROOT_DIR, SAVE_DIR, "loss_plot_varying_layers_2000_test_run_zeroth_scaled_1e7.png"))

ax.set_yscale("log")

plt.savefig(os.path.join(ROOT_DIR, SAVE_DIR, "loss_plot_varying_layers_log_2000_test_run_zeroth_scaled_1e7.png"))
plt.close()