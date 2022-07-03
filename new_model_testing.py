import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from keras.preprocessing.sequence import pad_sequences

from tensorflow import keras
from tensorflow.keras import layers




from case_declining.utils.data_utils import index_word, word_index
max_input_length = 5
max_output_length = 4
encoding_length = 35
input_data_length = 254


def one_hot_encode(sequence, seq_len=3, enc_len=encoding_length):
    encoded = np.zeros((seq_len, enc_len), dtype=int)
    for pos, val in enumerate(sequence):
        try:
            encoded[pos, val] = 1
        except IndexError:
            continue
    return encoded


df = shuffle(pd.read_csv('datasets/new'))
# print(df)

input = np.zeros([input_data_length, max_input_length, encoding_length])
# output = np.zeros([train_data_length, max_output_length, encoding_length])

input[:, 0, :] = np.repeat(df['G'].to_numpy(), encoding_length, axis=0).reshape(input_data_length, encoding_length)
input[:, 1, :] = np.repeat(df['S'].to_numpy(), encoding_length, axis=0).reshape(input_data_length, encoding_length)
input[:, 2:, :] = np.array([one_hot_encode(p)
                            for p in pad_sequences([[word_index[s] for s in d] for d in df['X'].apply(str.lower).values.tolist()],
                                                   padding='post', maxlen=3).tolist()
                            ])
train_data_x = input[:200]
val_data_x = input[200:]

output = np.array([one_hot_encode(p, max_output_length)
                   for p in pad_sequences([[word_index[s] for s in d] for d in df['Y'].apply(str.lower).values.tolist()],
                                          padding='post', maxlen=4).tolist()
                   ])
train_data_y = output[:200]
val_data_y = output[200:]


# print(val_data_x.shape)
# print(val_data_y.shape)

embed_dim = 256
latent_dim = 512

source = keras.Input(shape=(None,), dtype='int64')



