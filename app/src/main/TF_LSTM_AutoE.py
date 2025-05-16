from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

class Encoder(Model):
    def __init__(self, enc_units):
        super(Encoder, self).__init__()
        # LSTM returning its hidden states and cell states
        self.lstm = LSTM(enc_units,
                        return_state=True,
                        name='encoder_lstm')

    def call(self, x, training=False):
        # x shape ~ (batch_size, timesteps, num_encoder_tokens)
        _, state_h, state_c = self.lstm(x, training=training)
        # return the states to feed into the decoder
        return [state_h, state_c]

class Decoder(Model):
    def __init__(self, dec_units, num_decoder_tokens):
        super(Decoder, self).__init__()
        # LSTM returns both sequences and states
        self.lstm = LSTM(dec_units,
                         return_sequences=True,
                         return_state=True,
                         name='decoder_lstm')
        # Final projection layer to vocab distribution
        self.dense = Dense(num_decoder_tokens,
                           activation="softmax",
                           name="decoder_dense")

    def call(self, x, states, training=False):
        # x shape: (batch_size, timesteps, num_decoder_tokens)
        lstm_out, state_h, state_c = self.lstm(x,
                                               initial_state=states,
                                               training=training)
        output = self.dense(lstm_out)
        return output, [state_h, state_c]

class Seq2SeqAutoencoder(Model):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super(Seq2SeqAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=False):
        enc_input, dec_input = inputs
        # Encode the input sequence to get states
        enc_states = self.encoder(enc_input,
                                  training=training)
        # Decode, starting from the encoder states
        dec_output, _ = self.decoder(dec_input,
                                     enc_states,
                                     training=training)
        return dec_output