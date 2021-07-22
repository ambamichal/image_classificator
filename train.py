#import plotly.graph_objects as go
#import plotly.offline as po
#from plotly.subplots import make_subplots
from datetime import datetime
#import pandas as pd
import argparse
import pickle
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
#import tensorflow.keras.applications.efficientnet as efn
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from architecture import models
#import wandb

print(f'Wersja tensorflow:{tf.__version__}')


ap = argparse.ArgumentParser()
ap.add_argument('-e', '--epochs', default=1, help='Określ liczbę epok', type=int)
args = vars(ap.parse_args())

MODEL_NAME = 'LeNet5'
LEARNING_RATE = 0.01
EPOCHS = args['epochs']
BATCH_SIZE = 32
INPUT_SHAPE = (224, 224, 3)
TRAIN_DIR = 'images/train'
VALID_DIR = 'images/valid'


#augumentacja danych
train_datagen = ImageDataGenerator(
    rotation_range=30,
    rescale=1. / 255.,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1. / 255.)

train_generator = train_datagen.flow_from_directory(
    directory=TRAIN_DIR,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='binary'
)

valid_generator = valid_datagen.flow_from_directory(
    directory=VALID_DIR,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='binary'
)


#model = efn.EfficientNetB0(include_top=False, weights='imagenet')


architectures = {MODEL_NAME: models.LeNet5}
architecture = architectures[MODEL_NAME](input_shape=INPUT_SHAPE)
model = architecture.build()

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

dt = datetime.now().strftime('%d_%m_%Y_%H_%M')
filepath = os.path.join('output', 'model_' + dt + '.hdf5')
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', save_best_only=True)

print('[INFO] Trenowanie modelu')

history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

print('[INFO] Eksport etykiet do pliku...')
with open(r'output/labels.pickle', 'wb') as file:
    file.write(pickle.dumps(train_generator.class_indices))

print('[INFO] Koniec')

# uruchomienie z konsoli:
# $ python train.py -e 20

#wandb.init(
#  project="tf_klasyfikator",
#  notes="wandb_test",
#  tags=["baseline", "paper1"],
#  config=config,
#)