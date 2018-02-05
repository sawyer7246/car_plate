import matplotlib

import dateutil.parser
import datetime
import matplotlib.dates as mdates

import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.contrib import learn

from pymongo import MongoClient
from bson.objectid import ObjectId

from sklearn.metrics import mean_squared_error

from nick.lstm import lstm_model
from nick.data_processing import generate_data, load_csvdata

LOG_DIR = './data/lstm_plate'
TIMESTEPS = 10
RNN_LAYERS = [{'num_units': 5}]
DENSE_LAYERS = [10, 10]
TRAINING_STEPS = 1
BATCH_SIZE = 100
PRINT_STEPS = TRAINING_STEPS / 100


# downloaded weather data from http://www.ncdc.noaa.gov/qclcd/QCLCD
def load_plate_frame(filename):
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_file_relative_path = '/data/train_20171215.txt'

    train = pd.read_table(abs_path + train_file_relative_path, engine='python')
    train.describe()

    actions1 = train.groupby(['date', 'day_of_week'], as_index=False)['cnt'].agg({'count1': np.sum})

    df = pd.DataFrame(actions1, columns=['date', 'day_of_week', 'count1'])
    return df.set_index('date')


# scale values to reasonable values and convert to float
data_plate = load_plate_frame("data/train_20171215.txt")
X, y = load_csvdata(data_plate, TIMESTEPS, seperate=False)

regressor = learn.SKCompat(learn.Estimator(
    model_fn=lstm_model(
        TIMESTEPS,
        RNN_LAYERS,
        DENSE_LAYERS
    ),
    model_dir=LOG_DIR
))

validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'],
                                                     every_n_steps=PRINT_STEPS,
                                                     early_stopping_rounds=1000)
regressor.fit(X['train'], y['train'],
              monitors=[validation_monitor],
              batch_size=BATCH_SIZE,
              steps=TRAINING_STEPS)

predicted = regressor.predict(X['test'])


# plot the data
all_dates = data_plate.index.get_values()

fig, ax = plt.subplots(1)
fig.autofmt_xdate()

predicted_values = predicted.flatten() #already subset
predicted_dates = all_dates[len(all_dates)-len(predicted_values):len(all_dates)]
predicted_series = pd.Series(predicted_values, index=predicted_dates)
plot_predicted, = ax.plot(predicted_series, label='predicted (c)')

test_values = y['test'].flatten()
test_dates = all_dates[len(all_dates)-len(test_values):len(all_dates)]
test_series = pd.Series(test_values, index=test_dates)
plot_test, = ax.plot(test_series, label='2015 (c)')

xfmt = mdates.DateFormatter('%b %d %H')
ax.xaxis.set_major_formatter(xfmt)


# X[['date','result']].to_csv('data/result.txt', index=False, header=False, sep='\t')
# ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d %H')
plt.title('PDX Weather Predictions for 2016 vs 2015')
plt.legend(handles=[plot_predicted, plot_test])
plt.show()