import re
from collections import defaultdict
import pickle
import yaml

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f) 

def get_stats(vocab):
    """
    Given a vocabulary (dictionary mapping words to frequency counts), returns a 
    dictionary of tuples representing the frequency count of pairs of characters 
    in the vocabulary.
    """
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    """
    Given a pair of characters and a vocabulary, returns a new vocabulary with the 
    pair of characters merged together wherever they appear.
    """
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def get_vocab(data):
    """
    Given a list of strings, returns a dictionary of words mapping to their frequency 
    count in the data.
    """
    vocab = defaultdict(int)
    for line in data:
        for word in line.split():
            vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab

def byte_pair_encoding(data, n):
    """
    Given a list of strings and an integer n, returns a list of n merged pairs
    of characters found in the vocabulary of the input data.
    """

    data = data.lower()
    data = ' '.join(data.split())
    data = re.split(r'[.?!]', data)
    vocab = get_vocab(data)
    for i in range(n):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)

    # Assign token IDs
    token_to_id = {token: i for i, token in enumerate(vocab.keys(), start=4)}  
    # Reserving 0-3 for special tokens
    special_tokens = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
    token_to_id.update(special_tokens)
    return vocab, token_to_id


if __name__ == "__main__":
    """ Load the text file, run the byte_pair_encoding algorithm, then save in pickle """
    config = load_config()

    with open(config['tokenizer']['document'], "r", encoding="utf-8") as file:
        text = file.read()

    bpe_vocab, token_to_id = byte_pair_encoding(text, config['tokenizer']['numb_pair_merges'])

    with open(config['tokenizer']['tokens_location'], "wb") as f:
        pickle.dump({"vocab": bpe_vocab, "token_to_id": token_to_id}, f)