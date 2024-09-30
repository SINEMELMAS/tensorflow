# tensorflow
The code builds a model to classify IMDb reviews as positive or negative using an embedding layer, an LSTM, and a sigmoid output layer. It trains the model on 10,000-word vocabulary reviews padded to 500 words for 5 epochs.
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
vocab_size = 10000
maxlen = 500

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

train_data = sequence.pad_sequences(train_data, maxlen=maxlen)
test_data = sequence.pad_sequences(test_data, maxlen=maxlen)

model = models.Sequential([
    layers.Embedding(vocab_size, 128, input_length=maxlen),
    layers.LSTM(64),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_data, train_labels, epochs=5, batch_size=64, validation_split=0.2)
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test accuracy:', test_acc)
