# %%
import pandas as pd
from IPython.display import display

red_wine = pd.read_csv('red-wine.csv')

# Create training and validation splits
df_train = red_wine.sample(frac=0.7, random_state=0)
df_valid = red_wine.drop(df_train.index)
display(df_train.head(4))

# Scale to [0, 1]
max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)

# Split features and target
X_train = df_train.drop('quality', axis=1)
X_valid = df_valid.drop('quality', axis=1)
y_train = df_train['quality']
y_valid = df_valid['quality']
# %%
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import tensorflow as tf

tf.random.set_seed(1)

early_stopping = callbacks.EarlyStopping(
    min_delta=1e-5, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)

model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=[11]),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1),
])
model.compile(
    optimizer='adam',
    loss='mae',
)

from tensorflow.keras.callbacks import TensorBoard

tensorboard_callback = TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    write_graph=True,
    update_freq='batch',  # 'epoch' or 'batch'
    embeddings_freq=0,
    embeddings_metadata=None
)

# %%
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=500,
    callbacks=[early_stopping, tensorboard_callback],
    # callbacks=[early_stopping], # put your callbacks in a list
    verbose=1,  # turn off training log
)
# from tensorflow import keras
# from tensorflow.keras import layers, callbacks
# import tensorflow as tf
#
# tf.random.set_seed(1)
#
# early_stopping = callbacks.EarlyStopping(
#     min_delta=1e-5,
#     patience=20,
#     restore_best_weights=True,
# )
#
# model = keras.Sequential([
#     layers.Dense(512, activation='relu', input_shape=[11]),
#     layers.Dense(512, activation='relu'),
#     layers.Dense(512, activation='relu'),
#     layers.Dense(1),
# ])
# model.compile(
#     optimizer='adam',
#     loss='mae',
# )
#
# from tensorflow.keras.callbacks import TensorBoard
#
# tensorboard_callback = TensorBoard(
#     log_dir='./logs',
#     histogram_freq=1,
#     write_graph=True,
#     update_freq='epoch',  # 'epoch' or 'batch'
#     embeddings_freq=0,
#     embeddings_metadata=None
# )
#
# class LossHistory(callbacks.Callback):
#     def on_train_begin(self, logs=None):
#         self.losses = []
#         self.val_losses = []
#
#     def on_epoch_end(self, epoch, logs=None):
#         self.losses.append(logs.get('loss'))
#         self.val_losses.append(logs.get('val_loss'))
#         self._update_tensorboard(epoch)
#
#     def _update_tensorboard(self, epoch):
#         summary_writer = tf.summary.create_file_writer('./logs')
#         with summary_writer.as_default():
#             tf.summary.scalar('loss', self.losses[-1], step=epoch)
#             tf.summary.scalar('val_loss', self.val_losses[-1], step=epoch)
#
# loss_history = LossHistory()
#
# # ...
#
# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_valid, y_valid),
#     batch_size=256,
#     epochs=500,
#     callbacks=[early_stopping, tensorboard_callback, loss_history],
#     verbose=1,
# )




history_df = pd.DataFrame(history.history)
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))
history_df.loc[:, ['loss', 'val_loss']].plot()
#
# %%
