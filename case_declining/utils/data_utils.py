import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


index_word = {
    1: 'о', 2: 'а', 3: 'в', 4: ' ', 5: 'и', 6: 'н', 7: 'р', 8: 'л', 9: 'і', 10: 'к', 11: 'е', 12: 'ч', 13: 'с',
    14: 'д', 15: 'т', 16: 'м', 17: 'й', 18: 'г', 19: 'у', 20: 'я', 21: 'п', 22: 'ь', 23: 'б', 24: 'ю', 25: 'х',
    26: 'ш', 27: 'ї', 28: 'є', 29: 'ц', 30: 'з', 31: 'ф', 32: 'щ', 33: 'ж', 34: '’', 35: "'"
}

word_index = {
    'о': 1, 'а': 2, 'в': 3, ' ': 4, 'и': 5, 'н': 6, 'р': 7, 'л': 8, 'і': 9, 'к': 10, 'е': 11, 'ч': 12, 'с': 13,
    'д': 14, 'т': 15, 'м': 16, 'й': 17, 'г': 18, 'у': 19, 'я': 20, 'п': 21, 'ь': 22, 'б': 23, 'ю': 24, 'х': 25,
    'ш': 26, 'ї': 27, 'є': 28, 'ц': 29, 'з': 30, 'ф': 31, 'щ': 32, 'ж': 33, '’': 34, "'": 35
}


class DataConverter:
    COLUMN_X = 'X'              # nominative case
    COLUMN_Y = 'Y'              # genitive case
    COLUMN_G = 'G'
    DELTA = 'delta'

    def __init__(self, train_data_path, test_data_path=None, ngram_factor=(3, 3, 0), char_level=True, tokeizer=None):
        self.ngram_factor = ngram_factor
        # self.tk = tokeizer or Tokenizer(char_level=char_level)
        self.train_data = shuffle(self.load_data(train_data_path))
        # self.test_data = self.load_data(test_data_path)

        # self.max_length = max([self.train_data[col].str.len().max() for col in [self.column_x, self.column_y]])
        self.max_length = 12
        # self.tk.fit_on_texts(self.get_column_values(self.train_data, self.COLUMN_X))
        # self.encoding_length = len(self.tk.index_word.keys())
        self.encoding_length = 35

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
        return df.drop_duplicates(subset=[self.column_x, self.column_y], ignore_index=True)
        # return df

    @staticmethod
    def get_column_values(data, column_name):
        return data[column_name].values.tolist()

    def get_string_encoded(self, string):
        ## data = ' '.join([n[-g:] for n, g in zip(string.split(), cls.NGRRAM_FACTOR)])
        ## seq = [word_index[s] for s in data]
        ## i = iter(seq)

        data = ' '.join([n[-g:] for n, g in zip(string.split(), self.ngram_factor)])
        # seq = self.tk.texts_to_sequences(data)
        seq = [word_index[s] for s in data]
        padded_seq = pad_sequences([seq], padding='post', maxlen=self.max_length).tolist()[0]
        return np.array(self.one_hot_encode(padded_seq))

    def get_column_encoded_data(self, data, column_name=None):
        column_name = column_name if column_name else self.column_x
        data[column_name] = data[column_name].apply(str.lower)
        # sequences = self.tk.texts_to_sequences(self.get_column_values(data, column_name))
        # sequences = [[word_index[s] for s in ' '.join([n[-g:] for n, g in zip(d.split(), self.ngram_factor)])]
        #              for d in self.get_column_values(data, column_name)]
        sequences = [[word_index[s] for s in d] for d in self.get_column_values(data, column_name)]
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
        decoded_str = "".join([index_word.get(np.argmax(vector), '') for vector in encrypted_sequence])
        if self.ngram_factor:
            return self.revert_ngram_factor(input_str, decoded_str).title()
        return decoded_str.title()

    def revert_ngram_factor(self, input_str, decoded_str):
        return ' '.join([x[:-g] + y if g else x + y
                         for x, y, g in zip(input_str.split(), decoded_str.split(), self.ngram_factor)])
