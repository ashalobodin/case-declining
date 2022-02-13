import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


class DataConverter:
    COLUMN_X = 'X'              # nominative case
    COLUMN_Y = 'Y'              # genitive case
    COLUMN_G = 'G'
    DELTA = 'delta'

    def __init__(self, train_data_path, test_data_path, ngram_factor, char_level=True, tokeizer=None):
        self.ngram_factor = ngram_factor
        self.tk = tokeizer or Tokenizer(char_level=char_level)
        self.train_data = shuffle(self.load_data(train_data_path))
        self.test_data = self.load_data(test_data_path)

        self.max_length = max([self.train_data[col].str.len().max() for col in [self.column_x, self.column_y]])
        self.tk.fit_on_texts(self.get_column_values(self.train_data, self.COLUMN_X))
        self.encoding_length = len(self.tk.index_word.keys())

    @property
    def column_x(self):
        return f'{self.COLUMN_X}_ngram' if self.ngram_factor else self.COLUMN_X

    @property
    def column_y(self):
        return f'{self.COLUMN_Y}_ngram' if self.ngram_factor else self.COLUMN_Y

    @property
    def input_shape(self):
        return self.max_length, self.encoding_length

    def load_data(self, filename):
        df = pd.read_csv(filename, names=[self.COLUMN_G, self.COLUMN_X, self.COLUMN_Y])
        if self.ngram_factor:
            df[self.DELTA] = [
                [len(y[i])-len(x[i])+self.ngram_factor[i] for i in range(3)]
                for y, x in zip(df[self.COLUMN_Y].str.split(), df[self.COLUMN_X].str.split())
            ]

            df[self.column_x] = df[self.COLUMN_X].apply(
                func=lambda row: ' '.join([n[-g:] for n, g in zip(row.split(), self.ngram_factor)]))
            df[self.column_y] = df.apply(
                func=lambda row: ' '.join([n[-d:] for n, d in zip(row[self.COLUMN_Y].split(), row[self.DELTA])]),
                axis=1
            )
        return df

    @staticmethod
    def get_column_values(data, column_name):
        return data[column_name].values.tolist()

    def get_string_encoded(self, string):
        data = ' '.join([n[-g:] for n, g in zip(string.split(), self.ngram_factor)])
        seq = self.tk.texts_to_sequences(data)
        padded_seq = pad_sequences([seq], padding='post', maxlen=self.max_length).tolist()[0]
        return np.array(self.one_hot_encode(padded_seq))

    def get_column_encoded_data(self, data, column_name=None):
        column_name = column_name if column_name else self.column_x
        sequences = self.tk.texts_to_sequences(self.get_column_values(data, column_name))
        padded = pad_sequences(sequences, padding='post', maxlen=self.max_length).tolist()
        return np.array([self.one_hot_encode(p) for p in padded])

    def one_hot_encode(self, sequence):
        encoded = np.zeros((self.max_length, self.encoding_length), dtype=int)
        for pos, val in enumerate(sequence):
            try:
                encoded[pos, val] = 1
            except IndexError:
                continue
        return encoded

    def one_hot_decode(self, encrypted_sequence, input_str):
        decoded_str = "".join([self.tk.index_word.get(np.argmax(vector), '') for vector in encrypted_sequence])
        if self.ngram_factor:
            return self.revert_ngram_factor(input_str, decoded_str).title()
        return decoded_str.title()

    def revert_ngram_factor(self, input_str, decoded_str):
        return ' '.join([x[:-g] + y if g else x + y
                         for x, y, g in zip(input_str.split(), decoded_str.split(), self.ngram_factor)])
