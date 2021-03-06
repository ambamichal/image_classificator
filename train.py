from datetime import datetime
import pandas as pd
import argparse
import os
import warnings
import plotly.graph_objects as go
import plotly.offline as po
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from architecture import models


print(f'Wersja tensorflow:{tf.__version__}')


ap = argparse.ArgumentParser()
ap.add_argument('-e', '--epochs', default=1, help='Określ liczbę epok', type=int)
args = vars(ap.parse_args())

MODEL_NAME = 'nazwa sieci'
LEARNING_RATE = 0.0001
EPOCHS = args['epochs']
BATCH_SIZE = 32
INPUT_SHAPE = (224, 224, 3)
TRAIN_DIR = 'images/train'
VALID_DIR = 'images/valid'


def plot_hist(history, filename):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    fig = make_subplots(rows=2, cols=1, subplot_titles=('Accuracy', 'Loss'))

    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['accuracy'], name='train_accuracy',
                             mode='markers+lines', marker_color='#f29407'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_accuracy'], name='valid_accuracy',
                             mode='markers+lines', marker_color='#0771f2'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['loss'], name='train_loss',
                             mode='markers+lines', marker_color='#f29407'), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_loss'], name='valid_loss',
                             mode='markers+lines', marker_color='#0771f2'), row=2, col=1)

    fig.update_xaxes(title_text='Liczba epok', row=1, col=1)
    fig.update_xaxes(title_text='Liczba epok', row=2, col=1)
    fig.update_yaxes(title_text='Accuracy', row=1, col=1)
    fig.update_yaxes(title_text='Loss', row=2, col=1)
    fig.update_layout(width=1400, height=1000, title=f"Metrics: {MODEL_NAME}")

    po.plot(fig, filename=filename, auto_open=False)


#augumentacja danych
train_datagen = ImageDataGenerator(
    rotation_range=30,
    rescale=1. / 255.,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1. / 255.)

train_data = train_datagen.flow_from_directory(
    directory=TRAIN_DIR,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='binary'
)

valid_data = valid_datagen.flow_from_directory(
    directory=VALID_DIR,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='binary'
)


architectures = {MODEL_NAME: models.Convnet2}
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
#checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', save_best_only=True)

print('[INFO] Trenowanie modelu')

history = model.fit_generator(
    generator=train_data,
    steps_per_epoch=train_data.samples // BATCH_SIZE,
    validation_data=valid_data,
    validation_steps=valid_data.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=ModelCheckpoint(filepath=filepath, monitor='val_accuracy', save_best_only=True)
)



print('[INFO] Eksport wykresu do pliku html...')
filename = os.path.join('output', 'report_' + dt + '.html')
plot_hist(history, filename=filename)



print('[INFO] Koniec')

# uruchomienie z konsoli:
# $ python train.py -e 100

