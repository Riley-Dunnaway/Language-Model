import pickle
from tokenizer import load_config

config = load_config()

file = open(config['tokens_location'],'rb')
tokens = pickle.load(file)
print(tokens)