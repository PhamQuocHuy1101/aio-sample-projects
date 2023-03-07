import re

def text_normailze(text):
    text = text.lower()
    text = re.sub(r'[^\w\s\n]', '', text)
    text = ' '.join(t for t in text.split(' ') if len(t) > 0)
    return text

if __name__ == '__main__':
    a = 'Phạm Quốc  Huy" ./ '
    print(text_normailze(a))