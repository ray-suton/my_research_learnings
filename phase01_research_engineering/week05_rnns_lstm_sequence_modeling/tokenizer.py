import torch


PAD_TOKEN = "<PAD>"
EOS_TOKEN = "<EOS>"


class CharVocabulary:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.tokens)}

    @classmethod
    def from_sentences(cls, sentences):
        chars = sorted({char for sentence in sentences for char in sentence})
        return cls([PAD_TOKEN, EOS_TOKEN] + chars)

    def __len__(self):
        return len(self.tokens)

    @property
    def pad_id(self):
        return self.token_to_id[PAD_TOKEN]

    @property
    def eos_id(self):
        return self.token_to_id[EOS_TOKEN]

    def encode_text(self, text):
        return [self.token_to_id[char] for char in text]

    def encode_example(self, sentence):
        return self.encode_text(sentence) + [self.eos_id]

    def decode_ids(self, ids, skip_pad=True, stop_at_eos=False):
        pieces = []
        for idx in ids:
            token = self.id_to_token[int(idx)]
            if token == PAD_TOKEN and skip_pad:
                continue
            if token == EOS_TOKEN and stop_at_eos:
                break
            pieces.append(token)
        return "".join(pieces)


def find_unique_chars(sentences):
    return CharVocabulary.from_sentences(sentences).tokens

def hashes(tokens):
    word_to_id = {}
    id_to_word = {}

    for i, word in enumerate(tokens):
        word_to_id[word] = i
        id_to_word[i] = word 
    
    return word_to_id, id_to_word


def make_input_target(encoded, sequence_length, pad_id):
    input_ids = encoded[:-1]
    target_ids = encoded[1:]
    if len(input_ids) > sequence_length:
        return None

    padding = sequence_length - len(input_ids)
    return input_ids + [pad_id] * padding, target_ids + [pad_id] * padding


def build_sequence_tensors(sentences, vocabulary=None, sequence_length=128):
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    vocabulary = vocabulary or CharVocabulary.from_sentences(sentences)
    inputs = []
    targets = []
    skipped = 0
    for sentence in sentences:
        encoded = vocabulary.encode_example(sentence)
        pair = make_input_target(encoded, sequence_length, vocabulary.pad_id)
        if pair is None:
            skipped += 1
            continue
        input_ids, target_ids = pair
        inputs.append(input_ids)
        targets.append(target_ids)

    if not inputs:
        raise ValueError("No examples fit within sequence_length")

    X = torch.tensor(inputs, dtype=torch.long)
    y = torch.tensor(targets, dtype=torch.long)
    return X, y, vocabulary, skipped


def transform_sentences(sentences, sequence_length=128):
    vocab = CharVocabulary.from_sentences(sentences)
    X, y, _, _ = build_sequence_tensors(sentences, vocab, sequence_length)
    input_tokens = [[vocab.id_to_token[int(idx)] for idx in row] for row in X]
    target_tokens = [[vocab.id_to_token[int(idx)] for idx in row] for row in y]
    return input_tokens, target_tokens
    

def input_and_outputs(sentences, sequence_length=128):
    X, y, vocabulary, _ = build_sequence_tensors(sentences, sequence_length=sequence_length)
    return X, y, vocabulary.token_to_id, vocabulary.id_to_token
    
