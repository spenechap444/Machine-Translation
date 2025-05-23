from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Concatenate

class Encoder(Model):
    def __init__(self, enc_units):
        super(Encoder, self).__init__()
        self.bi_lstm = Bidirectional(
            LSTM(
                enc_units,
                activation='tanh',
                recurrent_activation='sigmoid',
                return_state=True,
                name='encoder_lstm'
            ),
            merge_mode='concat',
            name='bidirectional_encoder'
        )
        self.concat_h = Concatenate(name='encoder_h_concat')
        self.concat_c = Concatenate(name='encoder_c_concat')
        # encoder capacity doubles in size (hidden and cell states)
        # due to concatenating fwd and bkwd states

    def call(self, x, training=False):
        # x.shape : (batch_size, timesteps, num_encoder_tokens)
        # bi_lstm returns [output, f_h, f_c, b_h, b_c]
        _, forward_h, forward_c, backward_h, backward_c = self.bi_lstm(x, training=training)
        state_h = self.concat_h([forward_h, backward_h])
        state_c = self.concat_C([forward_c, backward_c])

        return [state_h, state_c]

class Decoder(Model):
    def __init__(self, dec_units, num_decoder_tokens):
        super(Decoder, self).__init__()
        # Unidirectional LSTM for generation
        self.lstm = LSTM(
            dec_units,
            return_sequences=True,
            return_state=True,
            name='decoder_lstm'
        )
        # Final projection to target vocabulary
        self.dense = Dense(
            num_decoder_tokens,
            activation='softmax',
            name = 'decoder_dense'
        )

    def call(self, x, states, training=False):
        # x.shape : (batch_size, timesteps, num_decoder_toekns)
        lstm_out, state_h, state_c = self.lstm(x,
                                               initial_state = states,
                                               training=training)
        output = self.dense(lstm_out)
        return output, [state_h, state_c]

class Seq2SeqAutoencoder(Model):
    def __init__(self, encoder, decoder):
        super(Seq2SeqAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=False):
        enc_input, dec_input = inputs
        # Encoding using BiLSTM
        enc_states = self.encoder(enc_input, training=training)
        # Decode, seeding with encoder states
        dec_output, _ = self.decoder(dec_input, enc_states, training=training)
        return dec_output

