import re
from collections import OrderedDict, Counter
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import vocab

def text_normalize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s\n]', '', text)
    text = ' '.join(t for t in text.split(' ') if len(t) > 0)
    return text

def build_vocab(from_file):
    data = {}    
    with open(from_file) as f:
        total_words = text_normalize(f.read()).split(' ')
        data = Counter(total_words)
    sorted_by_freq_tuples = sorted(data.items(), key=lambda x: x[1], reverse=True)
    dic = OrderedDict(sorted_by_freq_tuples)
    dic = vocab(dic)
    
    dic.set_default_index(-1)
    dic.append_token('<s>')
    dic.append_token('</s>')
    return dic

class Prepocessing:
    def __init__(self, vocab):
        self.vocab = vocab
    
    def __call__(self, text):
        text = text_normalize(text)
        words = ['<s>'] + text.split(' ') + ['</s>']
        ids = [self.vocab[w] for w in words]
        return ids

class TrainDataset(Dataset):
    def __init__(self, df, tranform):
        self.df = df
        self.df.sentence = self.df.sentence.apply(tranform)
        self.df.next_sentence = self.df.next_sentence.apply(tranform)
        # self.tranform = tranform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.iloc[dix]
        return row.sentence, row.next_sentence

class TrainDataUtils:
    @staticmethod
    def load_data_loader(df_file, preprocess, batch_size, shuffle):
        df = pd.read_csv(df_file)
        dataset = TrainDataset(df, preprocess)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader
if __name__ == '__main__':
    a = 'Phạm Quốc  Huy" ./ '
    print(text_normalize(a))