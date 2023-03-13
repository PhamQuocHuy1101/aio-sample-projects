import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
sentences, next_sentences = [], []
for idx, row in df.iterrows():
    lines = row.content.splitlines()
    sentences.extend(lines[:-1])
    next_sentences.extend(lines[1:])

wdf = pd.DataFrame({
    'id': list(range(len(sentences))),
    'sentence': sentences,
    'next_sentence': next_sentences
})
wdf.to_csv(sys.argv[2], index=False)