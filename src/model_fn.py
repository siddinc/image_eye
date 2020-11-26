from datetime import datetime
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import (
  Dense,
  Input,
  BatchNormalization,
  Conv2D,
  Conv2DTranspose,
  Flatten,
  Reshape,
  LeakyReLU,
  Activation,
  MaxPooling2D,
  UpSampling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import numpy as np
from constants import (
  SAVE_MODEL_PATH,
  LOAD_MODEL_PATH,
  INIT_LR,
  EPOCHS
)
from utility_fn import normalize_latent_features


def poly_decay(epoch):
  maxEpochs = EPOCHS
  baseLR = INIT_LR
  power = 1.0
  alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
  return alpha


def save_model_handler(model):
  now = datetime.now()
  model_name_suffix = now.strftime('%d-%m-%Y-%H:%M:%S')
  save_model(model, SAVE_MODEL_PATH + '/model${}.h5'.format(model_name_suffix))


def load_model_handler(model_name):
  loaded_model = load_model(LOAD_MODEL_PATH + '/{}'.format(model_name))
  return loaded_model


def predict_image(model, image):
  latent_features = model.predict(image)
  latent_features = normalize_latent_features(latent_features)
  return latent_features


def build_model(width, height, depth, filters=(32, 64), latentDim=256):
  input_shape = (height, width, depth)
  inputs = Input(shape=input_shape)
  x = inputs

  for f in filters:
    x = Conv2D(f, (3,3), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(axis=-1)(x)

  volume_size = K.int_shape(x)
  x = Flatten()(x)
  latent = Dense(latentDim)(x)
  encoder = Model(inputs, latent, name='encoder')
  
  latent_inputs = Input(shape=(latentDim,))
  x = Dense(np.prod(volume_size[1:]))(latent_inputs)
  x = Reshape((volume_size[1], volume_size[2], volume_size[3]))(x)

  for f in filters[::-1]:
    x = Conv2DTranspose(f, (3,3), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(axis=-1)(x)

  x = Conv2DTranspose(depth, (3,3), padding='same')(x)
  outputs = Activation('sigmoid')(x)

  decoder = Model(latent_inputs, outputs, name='decoder')

  autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
  return (encoder, decoder, autoencoder)


def compile_model(model, optimizer='adam', loss='mse', metrics=['accuracy']):
  callbacks = [LearningRateScheduler(poly_decay)]
  opt = Adam(lr=INIT_LR)
  model.compile(optimizer=opt, loss=loss, metrics=metrics)


def train_model(model, x_train, y_train, x_test, y_test, no_of_epochs, batch_size, callbacks):
    r = model.fit(
	    x_train, x_train,
	    validation_data=(x_test, x_test),
	    epochs=no_of_epochs,
	    batch_size=batch_size,
	    callbacks=callbacks
    )
    return r


def evaluate_model(model, x_test, y_test, batch_size):
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
    print(loss, accuracy)