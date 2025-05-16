from tensorflow.keras.models import Model
from tensorflow.keras.layers import SimpleRNN, Dense

class EncoderRNN(Model):
    def __init__(self, enc_units):
        super(EncoderRNN, self).__init__()
        self.rnn = SimpleRNN(enc_units,
                             return_state=True,
                             name="encoder_rnn")

    def call(self, x, training=False):
        # x shape: (batch_size, timesteps, num_encoder_tokens)
        _, state_h = self.rnn(x, training=training)
        # returning the final hidden state to pass to the decoder
        return state_h

class DecoderRNN(Model):
    def __init__(self, dec_units, num_decoder_tokens):
        super(DecoderRNN, self).__init__()
        self.rnn = SimpleRNN(dec_units,
                             return_sequences=True,
                             return_state=True,
                             name = "decoder_rnn")
        self.dense = Dense(num_decoder_tokens,
                           activation='softmax',
                           name='decoder_dense')

    def call(self, x, state, training=False):
        # x shape: (batch_size, timesteps, num_decoder_tokens)
        seq_output, state_h = self.rnn(x,
                                       initial_state=state,
                                       training=training)
        output = self.dense(seq_output)
        # returning the sequence predictions and new state
        return output, state_h

class Seq2SeqRNN(Model):
    def __init__(self, encoder: EncoderRNN, decoder: DecoderRNN):
        super(Seq2SeqRNN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=False):
        enc_input, dec_input = inputs
        # Encoding the input sequence to get input for decoder
        enc_state = self.encoder(enc_input,
                                 training=training)
        # Decoder returning predicted target sequences
        dec_output, _ = self.decoder(dec_input,
                                     enc_state,
                                     training=training)

        return dec_output