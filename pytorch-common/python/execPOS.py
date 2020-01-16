# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import re

torch.manual_seed(1)


######################################################################
# Example: An LSTM for Part-of-Speech Tagging
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In this section, we will use an LSTM to get part of speech tags. We will
# not use Viterbi or Forward-Backward or anything like that, but as a
# (challenging) exercise to the reader, think about how Viterbi could be
# used after you have seen what is going on.
#
# The model is as follows: let our input sentence be
# :math:`w_1, \dots, w_M`, where :math:`w_i \in V`, our vocab. Also, let
# :math:`T` be our tag set, and :math:`y_i` the tag of word :math:`w_i`.
# Denote our prediction of the tag of word :math:`w_i` by
# :math:`\hat{y}_i`.
#
# This is a structure prediction, model, where our output is a sequence
# :math:`\hat{y}_1, \dots, \hat{y}_M`, where :math:`\hat{y}_i \in T`.
#
# To do the prediction, pass an LSTM over the sentence. Denote the hidden
# state at timestep :math:`i` as :math:`h_i`. Also, assign each tag a
# unique index (like how we had word\_to\_ix in the word embeddings
# section). Then our prediction rule for :math:`\hat{y}_i` is
#
# .. math::  \hat{y}_i = \text{argmax}_j \  (\log \text{Softmax}(Ah_i + b))_j
#
# That is, take the log softmax of the affine map of the hidden state,
# and the predicted tag is the tag that has the maximum value in this
# vector. Note this implies immediately that the dimensionality of the
# target space of :math:`A` is :math:`|T|`.
#
#
# Prepare data:
def padIdxs(seq, MAX_LEN):
    padded = [0 for _ in range(MAX_LEN)]
    if len(seq) <= MAX_LEN:    
        padded[:len(seq)]=seq
    else: # should not happen
        print(len(seq)+"is longer than the max length : "+MAX_LEN)
    return padded

def prepare_sequence(seq, to_ix, MAX_LEN):
    idxs = [to_ix[w] for w in seq.split()]
    idxs=padIdxs(idxs, MAX_LEN)
    return torch.tensor(idxs, dtype=torch.long)
    
def load_datasets():
    text = data.Field(include_lengths=True)
    tags = data.Field()
    train_data, val_data, test_data = data.TabularDataset.splits(path='../data/RNN_Data_files/', train='train_data.tsv', validation='val_data.tsv', test='val_data.tsv', fields=[('text', text), ('tags', tags)], format='tsv')
    batch_sizes = (args.batch_size, args.batch_size, args.batch_size)
    train_loader, val_loader, test_loader = data.BucketIterator.splits((train_data, val_data, test_data), batch_sizes=batch_sizes, sort_key=lambda x: len(x.text))
    text.build_vocab(train_data)
    tags.build_vocab(train_data)
    dataloaders = {'train': train_loader,
                    'validation': val_loader,
                    'test': val_loader}
    return text, tags, dataloaders

def prepare_training_data(path_to_training_tsv):
    data = []
    with open(path_to_training_tsv,'r') as fp:
        lines = fp.readlines()
        for line in lines:
            words = line.split('\t')[0]
            tags = line.split('\t')[1]
            data.append((words.split(), tags.split()))
    return data


    

#pad training data


# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

######################################################################
# Create the model:


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores



class LSTMTaggerTS(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, batch_size):
        super(LSTMTaggerTS, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size=batch_size

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim,padding_idx=0)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, self.batch_size, self.hidden_dim),
                torch.zeros(1, self.batch_size, self.hidden_dim))
    def forward(self, sentences, input_lengths):
        self.hidden = self.init_hidden()
        embeds = self.word_embeddings(sentences)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeds, input_lengths, batch_first=True)
        lstm_out, self.hidden = self.lstm(packed, self.hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out,batch_first=True)
        outputs = outputs.contiguous()
        outputs = outputs.view(-1, outputs.shape[2])
        tag_space = self.hidden2tag(outputs)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

######################################################################
# Train the model:
#sent_batch = [training_data[i][0] for i in range(len(training_data))]
#batch_input_length=torch.LongTensor([ len(sent_batch[i]) for i in range(len(sent_batch))])
#MAX_LEN=max([ len(sent_batch[i]) for i in range(len(sent_batch))])


#TSmodel = LSTMTaggerTS(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), len(sent_batch))

#batch_inputs = prepare_sequence_batch(sent_batch, word_to_ix,MAX_LEN)



#TSmodel.eval()



##SORT
#batch_input_length, perm_idx = batch_input_length.sort(0,descending=True)
#batch_inputs = batch_inputs[perm_idx]

#traced_model = torch.jit.trace(TSmodel, (batch_inputs,batch_input_length))

#traced_model.save('tagger.pt')



def initDic(path_to_tsv): ## we only return word-to-ix for now! 
       #training_data = prepare_training_data('data/RNN_Data_files/train_data.tsv')
#    print("init Dic Called ! with : "+path_to_tsv)
    training_data = prepare_training_data(path_to_tsv)
    word_to_ix = {}
    word_to_ix['<PAD>']=0
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        #print(word_to_ix)
    tag_to_ix = {}
    tag_to_ix['<PAD>']=0
    for sent, tags in training_data:
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

    return word_to_ix

def preprocessData(strings, dic, MAX_LEN): #input numpy array of strings, dic used for preprocessing, max length of sentence , returns tensor
    
#    batch_inputs = prepare_sequence_batch(strings, dic, MAX_LEN)

    seq_batch=[]
    batch_len = []
    for seq in strings:
        #idxs=prepare_sequence(seq,dic, MAX_LEN)
        idxs = [dic[w] for w in seq.split()]
        batch_len.append(len(idxs))
        idxs = padIdxs(idxs, MAX_LEN)
        idxs = torch.tensor(idxs, dtype=torch.long)
        seq_batch.append(idxs)
    batch_inputs = torch.cat(seq_batch).view(len(strings), MAX_LEN)
    batch_input_length=torch.LongTensor([ batch_len[i] for i in range(len(strings))])
    batch_input_length, perm_idx = batch_input_length.sort(0,descending=True)
    batch_inputs = batch_inputs[perm_idx]
    return (batch_inputs, batch_input_length)
    
def prepare_sequence_batch(seqs, to_ix, MAX_LEN):
    seq_batch = []
    for seq in seqs:
        #seq = re.sub('[!?.@#$]', '', seq)
        idxs=prepare_sequence(seq,to_ix, MAX_LEN)
        seq_batch.append(idxs)        
    return torch.cat(seq_batch).view(len(seqs), MAX_LEN)

