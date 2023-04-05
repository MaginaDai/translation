import random, time
import torch
from torch import nn, Tensor
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
import math

from preprocessing import MAX_LENGTH
from utility import tensorsFromPair, test, timeSince, SOS_token, EOS_token


teacher_forcing_ratio = 0.5

class EncoderTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, d_model, n_head, num_layers, model_type, device, dropout=0.5):
        super(EncoderTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.d_model = d_model
        self.model_type = model_type
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src, src_mask):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, model_type, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.device = device

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.model_type = model_type

        if self.model_type == 'lstm':
            self.lstm = nn.LSTM(hidden_size, hidden_size)
        elif self.model_type == 'bi-lstm':
            self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional=True)
        else:
            self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        # output, hidden = self.gru(output, hidden)
        if self.model_type == 'lstm' or self.model_type == 'bi-lstm':
            output, hidden = self.lstm(output, hidden)
        else:
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        if self.model_type == 'lstm':
            return (torch.zeros(1, 1, self.hidden_size, device=self.device), torch.zeros(1, 1, self.hidden_size, device=self.device))
        elif self.model_type == 'bi-lstm':
            return (torch.zeros(2, 1, self.hidden_size, device=self.device), torch.zeros(2, 1, self.hidden_size, device=self.device))
        else:
            return torch.zeros(1, 1, self.hidden_size, device=self.device)
        
class Attention(nn.Module):
    def __init__(self, device):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.device = device
    
    def forward(self, hidden, encoder_hiddens):
        attention_scores = torch.matmul(hidden[0], encoder_hiddens.T)
        dist = self.softmax(attention_scores)
        output = torch.matmul(dist, encoder_hiddens)
        return output
        
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, model_type, device):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.model_type = model_type
        if self.model_type == 'lstm':
            self.lstm = nn.LSTM(hidden_size, hidden_size)
        elif self.model_type == 'bi-lstm':
            self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        else:
            self.gru = nn.GRU(hidden_size, hidden_size)
        
        self.attention = Attention(device)

        if self.model_type == 'bi-lstm' or self.model_type == 'attention':
            self.out = nn.Linear(hidden_size * 2, output_size)
        else:
            self.out = nn.Linear(hidden_size, output_size)
        
        self.softmax = nn.LogSoftmax(dim=1)

        self.device = device

    def forward(self, input, hidden, encoder_outputs):

        # Your code here #
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        if self.model_type == 'lstm':
            output, hidden = self.lstm(output, hidden)
        else:
            output, hidden = self.gru(output, hidden)
        
        if self.model_type == 'attention':
            attent_output = self.attention(hidden, encoder_outputs)
            output = torch.concat((attent_output, hidden[0]), dim=1)
            
        
        if self.model_type == 'bi-lstm':
            output = self.softmax(self.out(output[0]))
        elif self.model_type == 'bi-lstm':
            output = self.softmax(self.out(output))
        else:
            output = self.softmax(self.out(output))
        return output, hidden

    def initHidden(self):
        if self.model_type == 'lstm':
            return (torch.zeros(1, 1, self.hidden_size, device=self.device), torch.zeros(1, 1, self.hidden_size, device=self.device))
        else:
            return torch.zeros(1, 1, self.hidden_size, device=self.device)
        
    
def trainIters(input_lang, output_lang, train_pairs, test_pairs, encoder, decoder, epochs, model_type, device, logging, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    iter = 1
    n_iters = len(train_pairs) * epochs

    for epoch in tqdm(range(epochs)):
        # print("Epoch: %d/%d" % (epoch, epochs))
        logging.info("Epoch: %d/%d" % (epoch, epochs))
        for training_pair in train_pairs:
            training_pair = tensorsFromPair(input_lang, output_lang, training_pair, device)

            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = train(input_tensor, target_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, criterion, model_type, device)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                # print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                #                             iter, iter / n_iters * 100, print_loss_avg))
                logging.info('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

            iter +=1

        input,gt,predict,score = test(input_lang, output_lang, encoder, decoder, test_pairs, model_type, device)

        logging.info(f'Test Rouge1 f/p/r: {score["rouge1_fmeasure"]: .4f}/{score["rouge1_precision"]: .4f}/{score["rouge1_recall"]: .4f}')
        logging.info(f'Test Rouge2 f/p/r: {score["rouge2_fmeasure"]: .4f}/{score["rouge2_precision"]: .4f}/{score["rouge2_recall"]: .4f}')


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, model_type, device, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    if model_type == 'bi-lstm':
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size * 2, device=device)
    else:
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_hidden

    decoder_input = torch.tensor([[SOS_token]], device=device)

    if model_type == 'bi-lstm':
        decoder_hidden = encoder_hidden[0]
    else:
        decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length