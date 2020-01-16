'''
implement CBOW model with KJV bible dataset

code adopted from 
https://towardsdatascience.com/understanding-feature-engineering-part-4-deep-learning-methods-for-text-data-96c44370bbfa
'''

from nltk.tokenize.simple import SpaceTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import gutenberg
from model import CBOWModel

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
import numpy as np
import re
import sys
import time

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

def normalize(line):
    stemmer = PorterStemmer()
    stop_words = nltk.corpus.stopwords.words('english')

    res = []    

    for word in line:
        word = re.sub(r'[^a-zA-Z\s]', '', word)
        word = word.lower()    
        word = word.strip()    
        
        if word == "" or word in stop_words:
            pass 
        else:
            res.append(stemmer.stem(word))
    
    return " ".join(res)

class CBOWDataset(Dataset):
    def __init__(self, x, y):
        '''
        args : npndarray of x and y
        return : dataLoader
        '''
        self.x = torch.from_numpy(x).type(torch.LongTensor)
        self.y = torch.from_numpy(y).type(torch.LongTensor)

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


def generate(sents, c, dictionary):
    '''
    args : npndarray of strings
    return : context-target pair as list
    '''
    
    tokenizer = SpaceTokenizer()
    
    xs=[]
    ys=[]

    for sent in sents:
        sent = tokenizer.tokenize(sent)
        start = c
        end = len(sent)

        for i in range(start, end - c):
            context=[]
            for j in range(-c, c+1):
                if j==0: pass
                else:
                    context.append(dictionary.word2idx[sent[i+j]])
            
            xs.append(context)
            ys.append(dictionary.word2idx[sent[i]])

    x = np.vstack(xs)
    y = np.vstack(ys)
    
    return x, y
 
def batchify(pairs):
    '''
    args: generated pairs (list of lists)
    return: x npndarray, y npndarray
    '''   

    xs = []
    ys = []

    for n in range(0, len(pairs)):
        xs.append(pairs[n][0])
        ys.append(pairs[n][1])

    x = np.vstack(xs)
    y = np.vstack(ys)
    
    return x, y

def train(): # CPU training
    #import pdb; pdb.set_trace()
    model.train()
    total_loss = 0    

    for i, (x, y) in enumerate(dataloader):

        # forward
        out = model(x)
        y = y.view(-1)

        # loss and backward
        loss = criterion(out, y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        # logging
        if (i % log_interval) == 0 and i != 0:
            #TODO :
            # 1) implement training function
            print("batch_idx {:d} | loss {:.4f}".format(i, total_loss / log_interval))

            # 2) wrap up preprocessing -> create dataloader in certain form
            total_loss = 0
        


if __name__ == "__main__":
    tokenizer = SpaceTokenizer()
    normalize_corpus = np.vectorize(normalize)
    raw = gutenberg.sents('bible-kjv.txt')

    start_time = time.time()
    norm = normalize_corpus(raw[:100])   
    elapsed = time.time() - start_time

    # fill out dictionary
    dictionary = Dictionary()
    for sent in norm:
        words = tokenizer.tokenize(sent)
        for word in words:
            dictionary.add_word(word)
    '''
    print("length of dict: ", len(dictionary))
    print("word2idx: ", dictionary.word2idx)
    print("idx2dict: ", dictionary.idx2word)  
    '''

    # generate pairs
    start_time = time.time()
    pairs = generate(norm, 2, dictionary)
    elapsed = time.time() - start_time
   
    x, y = batchify(pairs)
    print(x[:10])
    print(y[:10])
    
    sys.exit(0)    

    dataset = CBOWDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = CBOWModel(len(dictionary), 100)
    
    # variables
    log_interval = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.01)  

    for epoch in range(10):
        train()

