import numpy as np
import pandas as pd
import os
import time
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Dropout, Bidirectional, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import keras_tuner as kt
from preprocessing import load_data, compute_features

# Paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT, "models")

# Data Loading
df = load_data()
df = compute_features(df)

label_encoder = LabelEncoder()
df['name_encoded'] = label_encoder.fit_transform(df['name'])
label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label map:", label_map)

features = ['wavenumbers', 'transmittance', 'gradient_transmittance', 'curvature_transmittance']
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Sequence Generation
SEQ_LEN = 200
X = df[features].values
y = df['name_encoded'].values

def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X, y, SEQ_LEN)
y_seq_cat = to_categorical(y_seq)
num_classes = y_seq_cat.shape[1]
input_shape = (SEQ_LEN, X_seq.shape[2])

X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq_cat, test_size=0.25, random_state=42)

# Custom Attention Layer
import tensorflow as tf
class SimpleAttention(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1), initializer='glorot_uniform', trainable=True)
    def call(self, inputs):
        scores = tf.matmul(inputs, self.W)
        weights = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(weights * inputs, axis=1)

# Model Builder
def build_model(input_shape, num_classes, hp=None):
    inputs = Input(shape=input_shape)
    filters1 = hp.Int('filters1', 16, 64, step=16) if hp else 32
    filters2 = hp.Int('filters2', 32, 128, step=32) if hp else 64
    lstm_units = hp.Int('lstm_units', 50, 150, step=25) if hp else 100
    dropout = hp.Float('dropout_rate', 0.2, 0.5, step=0.1) if hp else 0.3
    l2_reg = hp.Choice('l2_reg', [1e-4, 1e-3, 1e-2]) if hp else 1e-4

    x = Conv1D(filters1, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Conv1D(filters2, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = SimpleAttention()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# Tuner 
class CustomTuner(kt.Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        kwargs['callbacks'] = [EarlyStopping(monitor='val_loss', patience=5)]
        super().run_trial(trial, *args, **kwargs)

try:
    shutil.rmtree("tuner_dir")
except:
    pass

tuner = CustomTuner(
    lambda hp: build_model(input_shape, num_classes, hp),
    objective="val_accuracy",
    max_epochs=30,
    factor=3,
    directory="tuner_dir",
    project_name="ftir_dl"
)

tuner.search(X_train, y_train, validation_split=0.2, epochs=30)

# Final Model Train 
best_hp = tuner.get_best_hyperparameters(1)[0]
model = build_model(input_shape, num_classes, best_hp)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_hp.get("learning_rate", 1e-3)),
            loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=64,
        callbacks=[EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                    ModelCheckpoint(os.path.join(MODEL_PATH, "best_model.h5"), save_best_only=True)])

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"\nDL Test Accuracy: {test_acc*100:.2f}%")
