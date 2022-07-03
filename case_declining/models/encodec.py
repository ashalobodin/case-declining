from keras.layers import Dense, LSTM, Masking, Bidirectional, Embedding

from case_declining.models.common import CaseDecliningModelBase, default_optimizer


class EncoDec(CaseDecliningModelBase):
    def compile(self, hidden_units, activation, dropout=0., optimizer=default_optimizer, batch_size=1):
        # TODO: introduce different architecture/hidden_units for layers
        self.model.add(Masking(mask_value=0, input_shape=self.input_shape))
        self.model.add(Bidirectional(LSTM(hidden_units, return_sequences=True, dropout=dropout, unroll=True)))
        # self.model.add(Dense(self.input_shape[1]))
        self.model.add(Bidirectional(LSTM(hidden_units, return_sequences=True, dropout=dropout, unroll=True)))
        self.model.add(Dense(self.input_shape[1], activation=activation))
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
