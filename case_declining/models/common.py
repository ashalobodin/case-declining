from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import adam_v2

from nltk.translate import bleu
import numpy as np


default_optimizer = adam_v2.Adam(learning_rate=1e-3)


class CaseDecliningModelBase:

    SAVED_MODELS_DIR = 'saved_models'

    def __init__(self, model_name, input_shape, load=True, new_model_type=Sequential):
        self.input_shape = input_shape          # (max_length, encoding_length)
        self.model_name = f'{self.SAVED_MODELS_DIR}/{model_name}.h5'
        self.model = load_model(self.model_name, compile=False) if load else new_model_type()
        self.batch_size = 1

    @staticmethod
    def reshape_data(data):
        data = np.array(data)
        return data.reshape((1, *data.shape))

    def fit(self, train_x, train_y, validation_split, epochs=600, shuffle=True, verbose=2):
        history = self.model.fit(train_x, train_y, epochs=epochs, verbose=verbose, shuffle=shuffle,
                                 validation_split=validation_split)
        if verbose:
            print(self.model.summary())
            print([(k, v[-1]) for k, v in history.history.items()])
        return history.history

    def save(self, name=None):
        self.model.save(name or self.model_name)

    def predict(self, encoded_inp_value, verbose=0):
        if len(encoded_inp_value.shape) == 2:
            encoded_inp_value = self.reshape_data(encoded_inp_value)
        return self.model.predict(encoded_inp_value, verbose=verbose)

    # TODO: Introduce other score functions
    @staticmethod
    def bleu_score(references, hypothesis):
        return np.average(bleu([references], hypothesis, (1./3., )))

    def save_results(self):
        # TODO:
        pass
