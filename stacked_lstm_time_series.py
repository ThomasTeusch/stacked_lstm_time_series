#!/usr/local/bin/python3

# Import libraries
import sys
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from impyute.imputation.ts import locf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import copy
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional 
import keras
import tensorflow as tf

# Define environment variables for printing and plotting
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
pd.options.display.max_columns = 20
plt.style.use('ggplot')

# Define global variables
# Lag of previous data in input, LAG*15 min = previous data
LAG = 4*24*7  # = 7 days
P_TRAIN = 0.85  # Training set size
P_VAL = 0.10 # Validation set size
P_TEST = 1 - P_TRAIN - P_VAL # Test set size

# Import data from csv via pandas
inputpath = os.getcwd() + "/multivariate_time_series/"
inputfile = "dataset.csv"
df = pd.read_csv(inputpath+inputfile, sep=',')

time = df['time']
headers = [x for x in df.columns]
headers_no_time = copy.deepcopy(headers)
headers_no_time.remove('time')
N_FEATURES = len(headers_no_time)-1

# Fill missing values in "power" column with locf methode
imputation = True
if imputation:
    df = locf(np.asarray(
        df[headers_no_time]), axis=1)

df = pd.DataFrame(df, columns=headers_no_time)
df = pd.concat([time, df], axis=1)

# Feature Engineering
wind_vel = df.pop('wind_speed')
# Convert direction to radians.
wd_rad = df.pop('wind_direction')*np.pi / 180

# Calculate the wind x and y vectors.
# Solves two problems:
#   1. Wind directions of 0° and 359° should be close to each other
#   2. Wind directions should not matter if wind speed is near 0
df['wind_vec_x'] = wind_vel*np.cos(wd_rad)
df['wind_vec_y'] = wind_vel*np.sin(wd_rad)

column_names_inp = ['wind_vec_x', 'wind_vec_y', 'temperature', 'pressure']

# Quarterly (Q), Monthly (M), Weekly (W), Daily (D), Hourly (H), Minutely (M) aggregation of data
# df = df.resample('H').mean()

# Split data into training, validation and test set
n = len(df)
train_df = df[0:int(n * P_TRAIN )]
val_df = df[int(n* P_TRAIN ):int(n* (P_TRAIN + P_VAL))]
test_df = df[int(n*(P_TRAIN + P_VAL)):]
val_df.reset_index(inplace=True)
test_df.reset_index(inplace=True)

# Split input data and output
X_train = train_df[column_names_inp]
y_train = train_df['power']
X_val = val_df[column_names_inp]
y_val = val_df['power']
X_test = test_df[column_names_inp]
y_test = test_df['power']

# Standardize data according to new_val = (old_val - mean) / standard_deviation
X_scaler = StandardScaler()
Y_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
y_train = Y_scaler.fit_transform(y_train.values.reshape(-1, 1))
X_val = X_scaler.transform(X_val)
y_val = Y_scaler.transform(y_val.values.reshape(-1, 1))
X_test = X_scaler.transform(X_test)
y_test = Y_scaler.transform(y_test.values.reshape(-1, 1))

# Bring data into suitable form for LSTM
# ([samples, timesteps, features])
def make_time_series(X, y, lag, future_width):
    X_, y_ = [], []
    idx_range = range(len(X) - (lag + future_width))
    for idx in idx_range:
        X_.append(X[idx:idx+lag])
        y_.append(y[idx+lag+future_width - 1])
    X_ = np.array(X_)
    y_ = np.array(y_)
    return X_, y_

X_train, y_train = make_time_series(X=np.hstack((X_train, y_train)),
                                    y=y_train,
                                    lag=LAG, future_width=1)

X_val, y_val = make_time_series(X=np.hstack((X_val, y_val)),
                                y=y_val,
                                lag=LAG, future_width=1)

X_test, y_test = make_time_series(X=np.hstack((X_test, y_test)),
                                  y=y_test,
                                  lag=LAG, future_width=1)

y_train = y_train.reshape(-1,)

print(X_train.shape, y_train.shape)


# Initialize LSTM architecture
# Choose HeNormal as weight initialization
initializer = keras.initializers.HeNormal()

# Stacked LSTM architecture
# Three stacked LSTMs with dropout to prevent overfitting
inputs = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
lstm1 = keras.layers.LSTM(64,
                          kernel_initializer=initializer,
                          dropout = 0.2,
                          return_sequences=True
                        )(inputs)
lstm2 = keras.layers.LSTM(128,
                          kernel_initializer=initializer,
                          dropout = 0.2,
                          return_sequences=True
                         )(lstm1)
lstm3 = keras.layers.LSTM(64,
                          kernel_initializer=initializer,
                          dropout = 0.2,
                          return_sequences=False
                        )(lstm2)
dense1 = keras.layers.Dense(64)(lstm3)
drop1 = keras.layers.Dropout(0.2)(dense1)
outputs = keras.layers.Dense(1)(drop1)
model = keras.Model(inputs=inputs, outputs=outputs)

# Define RMSProp as optimizer using a learning rate of 0.01
LR = 0.01
opt = tf.keras.optimizers.RMSprop(learning_rate=LR)

# Compile LSTM net, choose MAE as loss function
model.compile(loss='mae', optimizer=opt, metrics=['mse'])
model.summary()

# Define Callbacks
# model_call: Save weights if validation_loss is decreasing
path_checkpoint = inputpath + "model_checkpoint.h5"
model_call = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

# early_call: stop training if validation_loss is not decreasing for 25 cycles
early_call = keras.callbacks.EarlyStopping(monitor="val_loss",
                                           min_delta=0, 
                                           patience=25,
                                           restore_best_weights=True)

# lr_call: decrease learning rate after 25 cycles exponentially
def scheduler(epoch, lr):
    if epoch < 25:
        return lr
    else:
        return lr * tf.math.exp(-0.01)
lr_call = keras.callbacks.LearningRateScheduler(scheduler)

training = True
if training:
    # Fit the modell
    hist = model.fit(X_train, y_train, 
                     epochs=200, 
                     batch_size=4*24*4,
                     validation_data=(X_val, y_val),
                     callbacks=[early_call, lr_call, model_call],
                     verbose=2,
                     shuffle=False)
    # Plot training results
    plt.plot(hist.history['loss'], label='train')
    plt.plot(hist.history['val_loss'], label='val')
    plt.legend()
    plt.show()
  
# Load model weights
model.load_weights(path_checkpoint)

# Make predictions with the trained model
def predict_and_rescale(data, x, model, scaler):
    if model is not None:
        data = model.predict(x)
    else:
        data = data.reshape(-1,1)  
    data = scaler.inverse_transform(data)
    return data.reshape(-1,1)

y_pred_train = predict_and_rescale(None, X_train, model, Y_scaler)
y_train = predict_and_rescale(y_train, None, None, Y_scaler)
y_pred_val = predict_and_rescale(None, X_val, model, Y_scaler)
y_val = predict_and_rescale(y_val, None, None, Y_scaler)
y_pred_test = predict_and_rescale(None, X_test, model, Y_scaler)
y_test = predict_and_rescale(y_test, None, None, Y_scaler)


# Calculate MAE and RMSE for Train, Val and Test
rmse_train = np.sqrt(mean_squared_error(y_pred_train, y_train))
mae_train = mean_absolute_error(y_pred_train, y_train)
rmse_val = np.sqrt(mean_squared_error(y_pred_val, y_val))
mae_val = mean_absolute_error(y_pred_val, y_val)
rmse_test = np.sqrt(mean_squared_error(y_pred_test, y_test))
mae_test = mean_absolute_error(y_pred_test, y_test)

# Print MAE and RMSE
print('Train RMSE: %.3f' % rmse_train)
print('Train MAE: %.3f' % mae_train)
print('Val RMSE: %.3f' % rmse_val)
print('Val MAE: %.3f' % mae_val)
print('Test RMSE: %.3f' % rmse_test)
print('Test MAE: %.3f' % mae_test)

# Plot predicted values vs true values
def plot(y1, y2, title):
  plt.plot(y1, label='Ground truth')
  plt.plot(y2, label='Prediction')
  plt.title(title)
  plt.legend()
  plt.show()

plot(y_train, y_pred_train, 'Training')
plot(y_val, y_pred_val, 'Validation')
plot(y_test, y_pred_test, 'Test')
