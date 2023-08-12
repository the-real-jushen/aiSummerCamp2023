# %% 
# Importings...
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

tf.random.set_seed(1)
# %% 
# Load data
stock_df = pd.read_csv('Google_Stock_Price.csv')
stock_df.head(10)
# %%
# Describe the data
stock_df.describe()
# %% 
# Check the data
stock_df.info()
# %% 
# Drop unecessary data
stock_df.drop('Date', axis=1, inplace=True)
# %% 
# Remove commas and change dtypes
stock_df.replace(regex=True,inplace=True,to_replace=r'\D',value=r'')

stock_df.astype('float64')
# %% 
# Split training and validation sets
stock_df_train = stock_df[:int(0.6*len(stock_df))]
stock_df_valid = stock_df[:int(0.6*len(stock_df)):int(0.8*len(stock_df))]
stock_df_test = stock_df[int(0.8*len(stock_df)):]
# %% 
# Normalization
scaler = MinMaxScaler()
scaler = scaler.fit(stock_df_train)
stock_df_train = scaler.transform(stock_df_train)
stock_df_valid = scaler.transform(stock_df_valid)
stock_df_test =  scaler.transform(stock_df_test)
# %% 
# Split X and y
def split_x_and_y(array, days_used_to_train=7):
    features = list()
    labels = list()

    for i in range(days_used_to_train, len(array)):
        features.append(array[i-days_used_to_train:i, :])
        labels.append(array[i, -1])
    return np.array(features), np.array(labels)

train_X, train_y = split_x_and_y(stock_df_train)
valid_X, valid_y = split_x_and_y(stock_df_valid)
test_X, test_y = split_x_and_y(stock_df_test)

print('Shape of Train X: {} \n Shape of Train y: {}'.format(train_X.shape, train_y.shape))
print(train_X[:5, -1, -1])
print(train_y[:5])
# %%
# Model establishing and compiling
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(units=64))
model.add(tf.keras.layers.Dense(1))

model.compile(
    optimizer='adam',
    loss='mse'
)

# %%
# Fitting
model.fit(
    train_X, train_y,
    validation_data=(valid_X, valid_y),
    batch_size=32,
    epochs=100
)
# %%
# Predicting
pred_y = model.predict(test_X)
# %% Plotting
plt.plot(range(len(pred_y)), pred_y, label='Prediction')
plt.plot(range(len(pred_y)), test_y, label='Ground Truth')
plt.xlabel('Amount of samples')
plt.ylabel('Prediction')
plt.legend()
plt.show()
# %%
