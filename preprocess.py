from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
import random
import pickle

class Tokenizer():
    def __init__(self):
        self.token2index = {}
        self.index2token = {}
        self.tokens = set()
        self.tokens.add('<unk>')
        self.tokens.add('<padding>')
        
    def __len__(self):
        return len(self.token2index)
        
    def vocab_dict(self, words):
        for word in words:
            self.tokens.add(word)
        for i, j in enumerate(self.tokens):
            self.token2index[j] = i
        for k, v in self.token2index.items():
            self.index2token[v] = k
            
    def text2sequence(self, sentence):
        sequence = []
        for i in sentence:
            sequence.append(self.token2index[i])
        return sequence
    
    def texts2sequences(self, sentences):
        sequences = []
        for i in sentences:
            sequences.append(self.text2sequence(i))
        return sequences
        
    def write(self, filepath):
        output = open(filepath, 'w')
        for k, v in self.token2index.items():
            print(k, v, file = output)
        output.close()
        
def load_w2v(filepath):
    w2v = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        tok_size, vec_size = map(int, lines[0].split())
        for line in lines[1:]:
            elements = line.split()
            w2v[elements[0]] = np.array(elements[1:], dtype = 'float64')
    return w2v, tok_size, vec_size
    
def load_annot_text(df, subject_id_names, hospital_id_names, target_names, text_names):
    texts = {}
    targets = {}
    s_ids = {}
    h_ids = {}
    for i in range(len(df)):
        s_ids[i] = df.loc[i, subject_id_names]
        h_ids[i] = df.loc[i, hospital_id_names]
        texts[i] = df.loc[i, text_names]
        current_targets = []
        for name in target_names:
            current_targets.append(df.loc[i, name])
        targets[i] = current_targets
    return texts, targets, s_ids, h_ids
    
#The regex cleaning function is predefined in the ipynb file to convert the text column to list of tokens
#for the training of word2vec.

class AnnotTextDataset(Dataset):
    def __init__(self, texts, targets, max_len, tokenizer):
        self.texts = texts
        self.targets = targets
        self.max_len = max_len
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        seq = self.tokenizer.text2sequence(text)
        if len(seq) < self.max_len:
            seq.extend([self.tokenizer.token2index['<padding>']]*(self.max_len - len(seq)))
        targets = self.targets[idx]
        return torch.LongTensor(seq), torch.LongTensor(targets)
            
class AnnotTextDataModule():
    def __init__(self, train_texts, val_texts, test_texts, train_targets, val_targets, test_targets, max_len, tokenizer, batchsize):
        self.batch_size = batchsize
        self.train = AnnotTextDataset(train_texts, train_targets, max_len, tokenizer)
        self.valid = AnnotTextDataset(val_texts, val_targets, max_len, tokenizer)
        self.test = AnnotTextDataset(test_texts, test_targets, max_len, tokenizer)
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size = self.batch_size)
        
    def val_dataloader(self):
        return DataLoader(self.valid, batch_size = self.batch_size)
        
    def test_dataloader(self):
        return DataLoader(self.test, batch_size = self.batch_size)

def getdata(annot_text_path, w2v_path, data_filename, batchsize, train_size, valid_size):
    
    # parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    # parser.add_argument('--annot_text_path', type = str, help = 'Path to annot_text.csv')
    # parser.add_argument('--w2v_path', type = str, help = 'Path to w2v.txt')
    # parser.add_argument('--batchsize', type = int, help = 'Batch size for training', default = 1)
    # parser.add_argument('--train_size', type = float, help = 'Percentage of training', default = 0.7)
    # parser.add_argument('--valid_size', type = float, help = 'Percentage of validation', default = 0.2)
    
    # args = parser.parse_args()
    
    print('Reading in {}'.format(annot_text_path.split('/')[-1]))
    df_annot_text = pd.read_pickle(annot_text_path)
    col_names = df_annot_text.columns
    text_names = 'text'
    hospital_id_names = 'Hospital_Admission_ID'
    subject_id_names = 'subject_id'
    target_names = col_names[2:-1]
    
    tokenizer = Tokenizer()
    all_words = []
    max_len = 0
    for i in range(len(df_annot_text)):
        words = df_annot_text.loc[i, text_names]
        all_words += words
        max_len = max(max_len, len(words))
        
    tokenizer.vocab_dict(all_words)
    print('vocabulary size is {}'.format(len(tokenizer.token2index)))
    print('num of all data points is {}'.format(df_annot_text.shape[0]))
    print('max sentence length is {}'.format(max_len))
    print('write text2index to dict_vocab.dict')
    tokenizer.write('dict_vocab.txt')
    
    w2v, tok_size, vec_size = load_w2v(w2v_path)
    texts, targets, subject_ids, hospital_ids = load_annot_text(df_annot_text, subject_id_names, hospital_id_names, target_names, text_names)
    
    #Make the embeddings matrix that will be used for the parameters of nn.embedding in the textCNN.
    unks = 0
    embeddings = np.random.uniform(-0.25, 0.25, (len(tokenizer.token2index), vec_size))
    for k, v in tokenizer.token2index.items():
        if k not in w2v.keys():
            if k == '<padding>':
                embeddings[v, :] = 0
            else:
                tokenizer.token2index[k] = tokenizer.token2index['<unk>']
                embeddings[v, :] = embeddings[tokenizer.token2index['<unk>'], :]
                unks += 1
        else:
            embeddings[v, :] = w2v[k]
    print('There are {} words that are not provided in the w2v table.'.format(unks))
    
    train_size = round(len(texts)*train_size)
    val_size = round(len(texts)*valid_size)
    test_size = len(texts) - train_size - val_size
    keys = list(texts.keys())
    random.shuffle(keys)
    train_keys, val_keys, test_keys = keys[:train_size], keys[train_size:train_size + val_size], keys[train_size + val_size:]
    train_texts = [texts[key] for key in train_keys]
    val_texts = [texts[key] for key in val_keys]
    test_texts = [texts[key] for key in test_keys]
    train_targets = [targets[key] for key in train_keys]
    val_targets = [targets[key] for key in val_keys]
    test_targets = [targets[key] for key in test_keys]
    all_targets = train_targets + val_targets + test_targets
    
    weights = []
    for n in range(len(target_names)):
        count_1 = 0
        for t in all_targets:
            if t[n] == 1:
                count_1 += 1
        weights_for_1 = len(all_targets)/(count_1*2)
        weights_for_0 = len(all_targets)/((len(all_targets) - count_1)*2)
        weights.append(torch.tensor([weights_for_0, weights_for_1]))
    
    with open(data_filename, "wb") as f:
        train_tar = np.array(train_targets)
        val_tar = np.array(val_targets)
        test_tar = np.array(test_targets)
        all_df = [train_texts, train_tar, val_texts, val_tar, test_texts, test_tar]
        pickle.dump(all_df, f)
    
    if batchsize == 0:
        return train_texts, train_tar, val_texts, val_tar, test_texts, test_tar
        
    else:
        ds = AnnotTextDataModule(train_texts, val_texts, test_texts, train_targets, val_targets, test_targets, max_len, tokenizer, batchsize)
        return ds.train_dataloader(), ds.val_dataloader(), ds.test_dataloader(), torch.from_numpy(embeddings), tokenizer.token2index['<padding>'], weights