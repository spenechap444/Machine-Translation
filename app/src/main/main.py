import Preprocessing as pre
import TF_LSTM_AutoE, TF_RNN_AutoE, TF_TeacherForcing
import os
from tensorflow.keras.callbacks import CSVLogger
import datetime


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
def train_TF_Autoencoder(model_type):

    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)) ,'data/cmn-eng', 'cmn.txt')) as f:
        data = f.read().split('\n')

    preprocesser = pre.Preprocess(num_sequences=100000)
    preprocesser.pad_sequences(data)
    encoder_input_data, decoder_input_data, decoder_target_data = preprocesser.one_hot_encode()



    if model_type == 'LSTM':
        print('Initiating LSTM model...')
        # hyperparameters
        latent_dim = 100
        batch_size = 64
        epochs = 50

        log_name = f'{model_type}_{str(datetime.datetime.now()).split(" ")[0]}_dim_{latent_dim}'

        # creating a csv logger for training results
        csv_logger = CSVLogger(os.path.join(os.path.dirname(__file__), 'train_results', f'{log_name}.csv'),
                               separator=',',
                               append=False)

       # num_encoder_tokens = len(preprocesser.input_characters)
        num_decoder_tokens = len(preprocesser.target_characters) # required for dense layer dim

        encoder = TF_LSTM_AutoE.Encoder(enc_units=latent_dim)
        decoder = TF_LSTM_AutoE.Decoder(dec_units=latent_dim, num_decoder_tokens=num_decoder_tokens)
        seq2seq = TF_LSTM_AutoE.Seq2SeqAutoencoder(encoder, decoder)

        seq2seq.compile(optimizer="rmsprop", loss="categorical_crossentropy")

        history = seq2seq.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                    batch_size = batch_size,
                    epochs=epochs,
                    validation_split=0.2,
                    callbacks=[csv_logger])
    elif model_type == 'RNN':
        print('Initiating RNN model...')
        # hyperparameters
        latent_dim = 100
        batch_size = 64
        epochs = 50

        # creating a csv logger for training results
        log_name = f'{model_type}_{str(datetime.datetime.now())}_dim_{latent_dim}'
        csv_logger = CSVLogger(os.path.join(os.path.dirname(__file__), 'train_results', f'{log_name}.csv'),
                               separator=',',
                               append=False)

        num_decoder_tokens = len(preprocesser.target_characters)

        encoder = TF_RNN_AutoE.EncoderRNN(enc_units=latent_dim)
        decoder = TF_RNN_AutoE.DecoderRNN(dec_units=latent_dim,
                                          num_decoder_tokens=num_decoder_tokens)
        seq2seq = TF_RNN_AutoE.Seq2SeqRNN(encoder, decoder)

        annealer = TF_TeacherForcing.AnnealTeacherForcing(seq2seq,
                                                          final_ratio=0.0,
                                                          epochs=50)

        seq2seq.compile(optimizer="rmsprop", loss="categorical_crossentropy")

        history = seq2seq.fit([encoder_input_data, decoder_input_data],
                    decoder_target_data,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.2,
                    callbacks=[annealer, csv_logger])

if __name__ == '__main__':
    # print(os.path.join(os.path.dirname(__file__), 'cmn-eng', 'cmn.txt'))
    # print_hi('PyCharm')
    train_TF_Autoencoder('RNN')

#########
# TO DO
# - Create a hyperparameter tuning framework
# - Get a larger training dataset ~ currently only using 10,000 records for training
# - Determine whether the autoencoder is undercomplete vs overcomplete, decide what to do
#########