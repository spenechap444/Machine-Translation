import numpy as np

class Preprocess:
    def __init__(self, num_sequences):
        self.num_sequences = num_sequences
        self.input_texts = []
        self.target_texts = []
        self.input_characters = set()
        self.target_characters = set()

    def _add_vocab_index(self, input_text, target_text):
        for char in input_text:
            if char not in self.input_characters:
                self.input_characters.add(char)
        for char in target_text:
            if char not in self.target_characters:
                self.target_characters.add(char)



    def pad_sequences(self, data):
        # checking max seq length in dataset against input param
        self.num_sequences = min(self.num_sequences, len(data)-1)
        for line in data[:self.num_sequences]:
            input_text, target_text, _ = line.split('\t') # _ is metadata
            # padding target sequences with start and end seq char
            target_text = '\t' + target_text + '\n'
            self.input_texts.append(input_text)
            self.target_texts.append(target_text)
            self._add_vocab_index(input_text, target_text)

        self.input_characters = sorted(list(self.input_characters))
        self.target_characters = sorted(list(self.target_characters))

    def one_hot_encode(self):
        # creating dictionaries to index each of the vocabularies
        input_token_index = dict([(char, i) for i, char in enumerate(self.input_characters)])
        target_token_index = dict([(char, i) for i, char in enumerate(self.target_characters)])

        # setting total vocab size for input and target datasets
        num_encoder_tokens = len(self.input_characters)
        num_decoder_tokens = len(self.target_characters)

        # setting max seq length for prior to one hot encoding
        max_encoder_seq_length = max([len(txt) for txt in self.input_texts])
        max_decoder_seq_length = max([len(txt) for txt in self.target_texts])

        # vector shape as (# of sentence pairs, max # seq len, # seq chars)
        encoder_input_data = np.zeros(
            (len(self.input_texts), max_encoder_seq_length, num_encoder_tokens),
            dtype='float32'
        )
        decoder_input_data = np.zeros(
            (len(self.input_texts), max_decoder_seq_length, num_decoder_tokens),
            dtype='float32'
        )
        decoder_target_data = np.zeros(
            (len(self.input_texts), max_decoder_seq_length, num_decoder_tokens),
            dtype='float32'
        )

        for i, (input_text, target_text) in enumerate(zip(self.input_texts, self.target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, input_token_index[char]] = 1.
            encoder_input_data[i, t+1:, input_token_index[' ']] = 1.
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by onetimestep
                decoder_target_data[i, t-1, target_token_index[char]] = 1.
                if t > 0:
                    # decoder_target data will not include start char since ahead by a step
                    decoder_target_data[i, t-1, target_token_index[char]] = 1.

            decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
            decoder_target_data[i, t:, target_token_index[' ']] = 1.

        return encoder_input_data, decoder_input_data, decoder_target_data

