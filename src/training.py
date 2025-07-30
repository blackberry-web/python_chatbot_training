import keras_preprocessing.text
import numpy as np
import tensorflow as tf
import keras
import keras_preprocessing
from preprocess import processed_data_questions, processed_data_answers
 
oov_tok = '<OOV>'
trunc_type='post'
padding_type='post'
max_length = 100
embedding_dim = 3
training_labels_size = len(processed_data_answers)
training_data_size = len(processed_data_questions)

tokenizer = keras_preprocessing.text.Tokenizer(oov_token=oov_tok)
tokenizer.fit_on_texts(processed_data_questions)
sequences = tokenizer.texts_to_sequences(processed_data_questions)
padded_sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

training_data = padded_sequences[:training_data_size]
training_labels = padded_sequences[:training_labels_size]
vocab_size = len(tokenizer.word_index) + 1

model = keras.Sequential([
    keras.layers.Embedding(training_data_size, embedding_dim),
    keras.layers.Dropout(0.2),
    keras.layers.Conv1D(64, 5, activation='relu'),
    keras.layers.MaxPooling1D(pool_size=4),
    keras.layers.LSTM(64),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(max_length, activation='softmax')
])
#training_size_labels
#sparse_categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#50
num_epochs = 2
history = model.fit(training_data, training_labels, epochs=num_epochs, verbose=2)