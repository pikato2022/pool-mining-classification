import os
import argparse
from datetime import datetime

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_directory',
    type=str,
    default='model',
    help='Directory to hold the model file.')
parser.add_argument(
    '--root_path',
    type=str,
    default='./',
    help='Root path of holding logs and trained model.')
args = parser.parse_args()

root_path = args.root_path
model_directory = args.model_directory

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

callbacks = [
  tf.keras.callbacks.TensorBoard(
      log_dir=root_path + '/logs/' + datetime.now().date().__str__()),
  tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
]

history = model.fit(x_train, y_train,
                    batch_size=32, epochs=5, callbacks=callbacks,
                    validation_data=(x_test, y_test))

output_path = os.path.join(root_path, model_directory)

model.save(output_path)