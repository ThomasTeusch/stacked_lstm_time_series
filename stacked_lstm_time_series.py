#!/usr/local/bin/python3

# Import all relevant python libraries
import os
import copy
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import matplotlib as mpl
from matplotlib import pyplot as plt
from impyute.imputation.ts import locf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set os, matplotlib, pyplot and pandas print options
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
plt.style.use('ggplot')
pd.options.display.max_columns = 20

# Define global variables for training
LAG = 16  # Lag of previous data in input, LAG*15 min = previous data
P_TRAIN = 0.9
P_VAL = 0.09
P_TEST = 1 - P_TRAIN - P_VAL

# Import data from csv via pandas (load for google Colab)
inputpath = "/content/drive/My Drive/Colab Notebooks/"
inputfile = "dataset.csv"
df = pd.read_csv(inputpath+inputfile, sep=',')

# Set variables for headers, number_of_features and
time = df['time']
headers = [x for x in df.columns]
headers_no_time = copy.deepcopy(headers)
headers_no_time.remove('time')
N_FEATURES = len(headers_no_time)-1


# Fill missing values with imputation method locf (time can't be handled)
df = locf(np.asarray(df[headers_no_time]), axis=1)

# Concate time column with other columns
df = pd.DataFrame(df, columns=headers_no_time)
df = pd.concat([time, df], axis=1)

# FEATURE ENGENEERING
#
# Build wind vectors from wind speed and wind_direction
# Deals with two problems:
#   1.) Direction 0° and 360° are now near each other
#   2.) Direction does not matter if wind speed equals 0
wind_vel = df.pop('wind_speed')
# Convert to radians.
wd_rad = df.pop('wind_direction')*np.pi / 180
# Calculate the wind x and y components.
df['wind_vec_x'] = wind_vel*np.cos(wd_rad)
df['wind_vec_y'] = wind_vel*np.sin(wd_rad)
# New column names
column_names_inp = ['wind_vec_x', 'wind_vec_y', 'temperature', 'pressure']

# TEST: Convert 15 min intervals into hourly or daily intervals
# Quarterly (Q), Monthly (M), Weekly (W), Daily (D), Hourly (H), Minutely (M)
# df = df.resample('H').mean()


# DATA preprocessing
# Build differencing method. Removes trends and seasonality
def difference(dataset, interval=1):
    diff = list()
    diff.append(np.nan)
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff

# Inverse differencing method.
def inverse_difference(last_ob, value):
    return value + last_ob

# Reshape data into format ([samples, timesteps, features])
def restructure_data(X, y, lag, future_width):
    X_, y_ = [], []
    idx_range = range(len(X) - (lag + future_width))
    for idx in idx_range:
        X_.append(X[idx:idx+lag])
        y_.append(y[idx+lag+future_width])
    X_ = np.array(X_)
    y_ = np.array(y_)
    return X_, y_


# Differenting 'Power' column
init_power = df['power'].head(1).values[0]
df['power'] = difference(df['power'])
df.dropna(inplace=True)
power_true = inverse_difference(init_power, df['power'])

# Train-Val-Test Split
n = len(df)
train_df = df[0:int(n * P_TRAIN)]
val_df = df[int(n * P_TRAIN):int(n * (P_TRAIN + P_VAL))]
test_df = df[int(n*(P_TRAIN + P_VAL)):]

val_df.reset_index(inplace=True)
test_df.reset_index(inplace=True)

# Split input data into input (X) and output(Y)
X_train = train_df[column_names_inp]
y_train = train_df['power']
X_val = val_df[column_names_inp]
y_val = val_df['power']
X_test = test_df[column_names_inp]
y_test = test_df['power']

# Scale data with StandardScaler (normalize data)
X_scaler = StandardScaler()
Y_scaler = StandardScaler()

X_train = X_scaler.fit_transform(X_train)
y_train = Y_scaler.fit_transform(y_train.values.reshape(-1, 1))
X_val = X_scaler.transform(X_val)
y_val = Y_scaler.transform(y_val.values.reshape(-1, 1))
X_test = X_scaler.transform(X_test)
y_test = Y_scaler.transform(y_test.values.reshape(-1, 1))

# Reshape data into format ([samples, timesteps, features])
X_train, y_train = restructure_data(X=X_train,
                                    y=y_train,
                                    lag=LAG, future_width=1)
X_val, y_val = restructure_data(X=X_val,
                                y=y_val,
                                lag=LAG, future_width=1)
X_test, y_test = restructure_data(X=X_test,
                                  y=y_test,
                                  lag=LAG, future_width=1)

# Save first value  from 'power' column in test set
first_test_val = np.array(power_true)[-len(y_test)]

# Set up LSTM
# Weight initialisation
initializer = keras.initializers.GlorotNormal()

# Define Stacked LSTM architecture
inputs = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
lstm_out1 = keras.layers.LSTM(128,
                              kernel_initializer=initializer,
                              return_sequences=True
                              )(inputs)
lstm_out2 = keras.layers.LSTM(256,
                              kernel_initializer=initializer,
                              return_sequences=True)(lstm_out1)
lstm_out3 = keras.layers.LSTM(512,
                              kernel_initializer=initializer,
                              return_sequences=True)(lstm_out2)
lstm_out4 = keras.layers.LSTM(256,
                              kernel_initializer=initializer
                              )(lstm_out3)
drop1 = keras.layers.Dropout(0.2)(lstm_out4)
dense1 = keras.layers.Dense(1024, activation='relu')(drop1)
outputs = keras.layers.Dense(1)(dense1)

model = keras.Model(inputs=inputs, outputs=outputs)

# Optimizer definition
LR = 0.01
opt = keras.optimizers.RMSprop(learning_rate=LR)
opt = keras.optimizers.SGD(learning_rate=LR)
#opt = keras.optimizers.Adadelta(learning_rate=LR)

# Compile LSTM net and print architecture
model.compile(loss='mse', optimizer=opt)
model.summary()

# Define Callbacks
# ModelCheckpoint Callback: Save best net weights
path_checkpoint = inputpath + "model_checkpoint.h5"
modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)
# EarlyStopping Callback: Stop training if no progress after 40 cycles
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss",
                                            min_delta=0, patience=40)

# LearningRate Callback: Reduce learning rate after 25 cycles exponentially


def scheduler(epoch, lr):
    if epoch < 25:
        return lr
    else:
        return lr * tf.math.exp(-0.01)


lr_callback = keras.callbacks.LearningRateScheduler(scheduler)

# Train LSTM network
training = True
if training:
    hist = model.fit(X_train, y_train, epochs=200, batch_size=LAG*24*4,
                     validation_data=(X_val, y_val),
                     callbacks=[es_callback, lr_callback, modelckpt_callback],
                     verbose=0,
                     shuffle=False)

# Load best weights
model.load_weights(path_checkpoint)

# plot history
plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='val')
plt.legend()
plt.show()

# Make prediction
y_pred = model.predict(X_test)
# Inverse transformation of 'power' column
y_pred = Y_scaler.inverse_transform(y_pred)
y_pred = inverse_difference(first_test_val, y_pred)
y_test = Y_scaler.inverse_transform(y_test)
y_test = inverse_difference(first_test_val, y_test)

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(y_pred, y_test))
mae = mean_absolute_error(y_pred, y_test)
print('Test RMSE: %.3f' % rmse)
print('Test MAE: %.3f' % mae)

# Plot prediction vs ground thruth
plt.plot(y_test, label='Ground Truth')
plt.plot(y_pred, label='Vorhersage')
plt.legend()
plt.show()
