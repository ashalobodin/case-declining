import numpy as np
import pandas as pd

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


class DataConverter:
    COLUMN_X = 'X'
    COLUMN_Y = 'Y'

    def __init__(self, train_data_path, validation_data_path, test_data_path, char_level=True):
        self.tk = Tokenizer(char_level=char_level)
        self.train_data = self.load_data(train_data_path)
        self.val_data = self.load_data(validation_data_path)
        self.test_data = self.load_data(test_data_path)

        self.max_length = max([self.train_data[col].str.len().max() for col in self.train_data.columns])
        self.tk.fit_on_texts(self.get_column_values(self.train_data))
        self.encoding_length = len(self.tk.index_word.keys())

    @property
    def input_shape(self):
        return self.max_length, self.encoding_length

    @classmethod
    def load_data(cls, filename):
        # X - nominative case
        # Y - genitive case
        return pd.read_csv(filename, names=[cls.COLUMN_X, cls.COLUMN_Y])

    @staticmethod
    def get_column_values(data, column=COLUMN_X):
        return data[column].values.tolist()

    def get_string_encoded(self, string):
        seq = self.tk.texts_to_sequences(string)
        padded_seq = pad_sequences([seq], padding='post', maxlen=self.max_length).tolist()[0]
        return np.array(self.one_hot_encode(padded_seq))

    def get_column_encoded_data(self, data, column=COLUMN_X):
        sequences_x = self.tk.texts_to_sequences(self.get_column_values(data, column))
        padded_x = pad_sequences(sequences_x, padding='post', maxlen=self.max_length).tolist()
        return np.array([self.one_hot_encode(p) for p in padded_x])

    def one_hot_encode(self, sequence):
        encoded = np.zeros((self.max_length, self.encoding_length), dtype=int)
        for pos, val in enumerate(sequence):
            try:
                encoded[pos, val] = 1
            except IndexError:
                continue
        return encoded

    def one_hot_decode(self, encrypted_sequence):
        return [self.tk.index_word.get(np.argmax(vector), '') for vector in encrypted_sequence]
