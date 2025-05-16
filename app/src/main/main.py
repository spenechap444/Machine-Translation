import Preprocessing as pre
import TF_LSTM_AutoE, TF_RNN_AutoE
import os

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
def train_TF_Autoencoder(model_type):

    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)) ,'cmn-eng', 'cmn.txt')) as f:
        data = f.read().split('\n')

    preprocesser = pre.Preprocess(num_sequences=10000)
    preprocesser.pad_sequences(data)
    encoder_input_data, decoder_input_data, decoder_target_data = preprocesser.one_hot_encode()


    if model_type == 'LSTM':
        # hyperparameters
        latent_dim = 25
        batch_size = 64
        epochs = 100

       # num_encoder_tokens = len(preprocesser.input_characters)
        num_decoder_tokens = len(preprocesser.target_characters) # required for dense layer dim

        encoder = TF_LSTM_AutoE.Encoder(enc_units=latent_dim)
        decoder = TF_LSTM_AutoE.Decoder(dec_units=latent_dim, num_decoder_tokens=num_decoder_tokens)
        seq2seq = TF_LSTM_AutoE.Seq2SeqAutoencoder(encoder, decoder)

        seq2seq.compile(optimizer="rmsprop", loss="categorical_crossentropy")

        seq2seq.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                    batch_size = batch_size,
                    epochs=epochs,
                    validation_split=0.2)
    elif model_type == 'RNN':
        # hyperparameters
        latent_dim = 50
        batch_size = 64
        epochs = 50

        num_decoder_tokens = len(preprocesser.target_characters)

        encoder = TF_RNN_AutoE.EncoderRNN(enc_units=latent_dim)
        decoder = TF_RNN_AutoE.DecoderRNN(dec_units=latent_dim,
                                          num_decoder_tokens=num_decoder_tokens)
        seq2seq = TF_RNN_AutoE.Seq2SeqRNN(encoder, decoder)

        seq2seq.compile(optimizer="rmsprop", loss="categorical_crossentropy")

        seq2seq.fit([encoder_input_data, decoder_input_data],
                    decoder_target_data,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.2)

if __name__ == '__main__':
    # print(os.path.join(os.path.dirname(__file__), 'cmn-eng', 'cmn.txt'))
    # print_hi('PyCharm')
    train_TF_Autoencoder('RNN')