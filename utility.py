import torch
import random

from preprocessing import MAX_LENGTH

import numpy as np
from torchmetrics.text.rouge import ROUGEScore
import time
import math

SOS_token = 0
EOS_token = 1

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


rouge = ROUGEScore()


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence, device):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(input_lang, output_lang, pair, device):
    input_tensor = tensorFromSentence(input_lang, pair[0], device)
    target_tensor = tensorFromSentence(output_lang, pair[1], device)
    return (input_tensor, target_tensor)


def evaluate(input_lang, output_lang, encoder, decoder, sentence, model_type, device, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        if model_type == 'bi-lstm':
            encoder_outputs = torch.zeros(max_length, encoder.hidden_size * 2, device=device)
        else:
            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        if model_type == 'bi-lstm':
            decoder_hidden = encoder_hidden[0]
        else:
            decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words


def evaluateRandomly(pairs, encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def test(inputlang, output_lang, encoder, decoder, testing_pairs, model_type, device):
    input = []
    gt = []
    predict = []
    metric_score = {
        "rouge1_fmeasure":[],
        "rouge1_precision":[],
        "rouge1_recall":[],
        "rouge2_fmeasure":[],
        "rouge2_precision":[],
        "rouge2_recall":[]
    }

    for pair in testing_pairs:
        output_words = evaluate(inputlang, output_lang, encoder, decoder, pair[0], model_type, device)
        output_sentence = ' '.join(output_words)

        input.append(pair[0])
        gt.append(pair[1])
        predict.append(output_sentence)
        metric_score["rouge1_fmeasure"].append(rouge(output_sentence, pair[1])['rouge1_fmeasure'])
        metric_score["rouge1_precision"].append(rouge(output_sentence, pair[1])['rouge1_precision'])
        metric_score["rouge1_recall"].append(rouge(output_sentence, pair[1])['rouge1_recall'])

        metric_score["rouge2_fmeasure"].append(rouge(output_sentence, pair[1])['rouge2_fmeasure'])
        metric_score["rouge2_precision"].append(rouge(output_sentence, pair[1])['rouge2_precision'])
        metric_score["rouge2_recall"].append(rouge(output_sentence, pair[1])['rouge2_recall'])

    metric_score["rouge1_fmeasure"] = np.array(metric_score["rouge1_fmeasure"]).mean()
    metric_score["rouge1_precision"] = np.array(metric_score["rouge1_precision"]).mean()
    metric_score["rouge1_recall"] = np.array(metric_score["rouge1_recall"]).mean()
        
    metric_score["rouge2_fmeasure"] = np.array(metric_score["rouge2_fmeasure"]).mean()
    metric_score["rouge2_precision"] = np.array(metric_score["rouge2_precision"]).mean()
    metric_score["rouge2_recall"] = np.array(metric_score["rouge2_recall"]).mean()

    print(f'Rouge1 f/p/r: {metric_score["rouge1_fmeasure"]: .4f}/{metric_score["rouge1_precision"]: .4f}/{metric_score["rouge1_recall"]: .4f}')
    print(f'Rouge2 f/p/r: {metric_score["rouge2_fmeasure"]: .4f}/{metric_score["rouge2_precision"]: .4f}/{metric_score["rouge2_recall"]: .4f}')
    return input,gt,predict,metric_score