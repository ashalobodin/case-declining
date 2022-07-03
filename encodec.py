import pandas as pd
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, TimeDistributed, RepeatVector, Masking
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.utils import to_categorical
from nltk.translate import bleu


tk = Tokenizer(char_level=True)


train_data = pd.read_csv('datasets/train_data.txt', names=['X', 'Y'])
data_length = len(train_data)

max_length = max([train_data[col].str.len().max() for col in train_data.columns])
train_x_values = train_data.X.values.tolist()
train_y_values = train_data.Y.values.tolist()

tk.fit_on_texts(train_x_values)
encoding_length = len(tk.index_word.keys())
print(tk.word_index)

print(max_length, encoding_length)


def one_hot_encode(seq, length=max_length, encoding_len=encoding_length):
    encoded = np.zeros((length, encoding_len), dtype=int)
    for pos, val in enumerate(seq):
        try:
            encoded[pos, val] = 1
        except IndexError:
            continue
    return encoded


def one_hot_decode(enc_seq):
    return [tk.index_word.get(np.argmax(vector), '') for vector in enc_seq]


sequences_x = tk.texts_to_sequences(train_x_values)
padded_x = pad_sequences(sequences_x, padding='post', maxlen=max_length).tolist()
train_x_encoded = np.array([one_hot_encode(p) for p in padded_x])


sequences_y = tk.texts_to_sequences(train_y_values)
padded_y = pad_sequences(sequences_y, padding='post', maxlen=max_length).tolist()
train_y_encoded = np.array([one_hot_encode(p) for p in padded_y])


model = Sequential()

# model 1
# model.add(Masking(mask_value=0, input_shape=(max_length, encoding_length)))
# model.add(LSTM(150, return_sequences=True))
# model.add(Dense(encoding_length, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model 2
model.add(Masking(mask_value=0, input_shape=(max_length, encoding_length)))
model.add(LSTM(100, return_sequences=True))
model.add(Dense(encoding_length))
model.add(LSTM(100, return_sequences=True))
model.add(Dense(encoding_length, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_name = 'vanilla_200ep'


# model.fit(train_x_encoded, train_y_encoded, epochs=600, shuffle=True)    # verbose=2,
# print(model.summary())
# model.save(f'../saved_models/{model_name}.h5')

# =================================================================
model = load_model(f'saved_models/{model_name}.h5')

# val_data = pd.read_csv('datasets/validation_data.txt', names=['X', 'Y'])
# val_data = pd.read_csv('datasets/small_train_data.txt', names=['X', 'Y'])
val_data = pd.read_csv('datasets/test_data.txt', names=['X', 'Y'])

train_v_values = val_data.X.values.tolist()
sequences_v = tk.texts_to_sequences(train_v_values)
padded_v = pad_sequences(sequences_v, padding='post', maxlen=max_length).tolist()
train_v_encoded = np.array([one_hot_encode(p) for p in padded_v])

val_data['Y'] = val_data['Y'].str.title()


def calc_averag_bleu(df):
    return np.average(bleu([list(df['Y'])], list(df['V']), (1./3., )))


print(train_v_encoded.shape)
yhat = model.predict(train_v_encoded, verbose=0)

# TODO: vectorize
val_data['V'] = [''.join(one_hot_decode(y)).title() for i, y in enumerate(yhat)]


print(val_data[['Y', 'V']])

print(calc_averag_bleu(val_data))
