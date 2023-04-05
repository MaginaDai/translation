import argparse
import logging
import os
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from model import Decoder, EncoderRNN, trainIters
from preprocessing import prepareData
from utility import test


SEED = 1234
hidden_size = 512
epochs = 20

parser = argparse.ArgumentParser(description='PyTorch Contrastive Learning for Wearable Sensing')

parser.add_argument('-lr', default=0.01, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-e', default=1, type=int, help='epoch')
parser.add_argument('-name', default='test', type=str, help='name of the model')
parser.add_argument('-model_type', default='attention', type=str, help='name of the model')

def main():
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    args = parser.parse_args()

    epochs = args.e
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

    X = [i[0] for i in pairs]
    y = [i[1] for i in pairs]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    train_pairs = list(zip(X_train,y_train))
    test_pairs = list(zip(X_test,y_test))
    model_type = args.model_type
    
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size, model_type, device).to(device)
    decoder1 = Decoder(hidden_size, output_lang.n_words, model_type, device).to(device)

    log_dir = f"e{args.e}_lr{args.lr}"
    if args.name:
        log_dir = args.name + '_' + log_dir
    log_dir = "runs/" + log_dir

    store_model_dir = log_dir + '/model.pt'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(filename=os.path.join(log_dir, 'training.log'), level=logging.DEBUG)

    trainIters(input_lang, output_lang, train_pairs, test_pairs, encoder1, decoder1, epochs, model_type, device, logging, print_every=5000, learning_rate=args.lr)
    torch.save({'encoder': encoder1.state_dict(), 'decoder': decoder1.state_dict()}, store_model_dir)

if __name__ == '__main__':
    main()