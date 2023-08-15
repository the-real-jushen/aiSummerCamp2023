# %%
# Importings
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve
tf.random.set_seed(1)
# %%
# Load data
df = pd.read_csv('ecg.csv', header=None)
raw_data = df.values
df.head()
# %%
# Split sets
labels = raw_data[:, -1]
features = raw_data[:, :-1]
X, test_X, y, test_y = train_test_split(
    features, labels, test_size=0.2, random_state=2
)
train_X, valid_X, train_y, valid_y = train_test_split(
    X, y, test_size=0.2, random_state=2
)
# %%
# Normalization to [0, 1]
scaler = MinMaxScaler()
scaler.fit(train_X)
train_X = scaler.transform(train_X)
valid_X = scaler.transform(valid_X)
test_X = scaler.transform(test_X)
# %% 
# Train the autoencoders with only normal rhythms
train_y = train_y.astype(bool)
test_y = test_y.astype(bool)

normal_train_X = train_X[train_y]
anomalous_train_X = train_X[~train_y]

normal_test_X = test_X[test_y]
anomalous_test_X = test_X[~test_y]
# %% 
# Plot a normal ECG
plt.grid()
plt.plot(range(len(normal_train_X[0])), normal_train_X[0])
plt.title("A Normal ECG")
plt.show()
# %% 
# Plot an anomalous ECG
plt.grid()
plt.plot(np.arange(140), anomalous_train_X[0])
plt.title("An Anomalous ECG")
plt.show()
# %% 
# Build the model
class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    
    self.encoder = tf.keras.Sequential([
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu")])

    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(140, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = AnomalyDetector()
autoencoder.compile(optimizer='adam', loss='mae')
# %% 
# Fitting
history = autoencoder.fit(normal_train_X, normal_train_X, 
          epochs=20, 
          batch_size=512,
          validation_data=(test_X, test_X),
          shuffle=True)
# %% 
# Plotting loss curves
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
# %% 
# Plotting reconstruction of normal data from test set
normal_encoded_data = autoencoder.encoder(normal_test_X).numpy()
normal_decoded_data = autoencoder.decoder(normal_encoded_data).numpy()

plt.plot(normal_test_X[0], 'b')
plt.plot(normal_decoded_data[0], 'r')
plt.fill_between(np.arange(140), normal_decoded_data[0], normal_test_X[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()
# %% 
# Plotting reconstruction of anomalous data from the test set
anomalous_encoded_data = autoencoder.encoder(anomalous_test_X).numpy()
anomalous_decoded_data = autoencoder.decoder(anomalous_encoded_data).numpy()

plt.plot(anomalous_test_X[0], 'b')
plt.plot(anomalous_decoded_data[0], 'r')
plt.fill_between(np.arange(140), anomalous_decoded_data[0], anomalous_test_X[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()
# %% 
# Detect anomalies
reconstructions = autoencoder.predict(normal_train_X)
train_loss = tf.keras.losses.mae(reconstructions, normal_train_X)

plt.hist(train_loss[None,:], bins=50)
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.show()
# %% 
# Determining threshold
threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)
# %% 
# Check anomalies reconstrution distribution on test set
reconstructions = autoencoder.predict(anomalous_test_X)
test_loss = tf.keras.losses.mae(reconstructions, anomalous_test_X)

plt.hist(test_loss[None, :], bins=50)
plt.xlabel("Test loss")
plt.ylabel("No of examples")
plt.show()
# %% 
# prediction and evaluation
def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return tf.math.less(loss, threshold)

prediction = predict(autoencoder, test_X, threshold)

fpr, tpr, _ = roc_curve(prediction, test_y)
print("TPR: {}, FPR: {}".format(tpr[1], fpr[1]))
# %%
