import re

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


def remove_stopword():
    stopword = []
    with open('.\\WordList\\stopword_list.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            stopword.append(line)
    return stopword


def remove_academic_word():
    academic_word = []
    with open('.\\WordList\\academic_word_list-2980.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            academic_word.append(line)
    return academic_word


def NLP_Prpcessing(text):
    text = str(text)
    text = re.sub(r'(https|http)://[a-zA-Z0-9.?/&=:]*', ' ', text)
    text = re.sub(r'[^a-zA-Z\']', ' ', text)
    text = ' '.join(text.split())
    text = text.lower()

    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)

    text = [token for token in text if not token.isnumeric()]
    text = [token for token in text if len(token) > 2]
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(token, pos='v') for token in text]
    text = [lemmatizer.lemmatize(token, pos='a') for token in text]
    text = [lemmatizer.lemmatize(token, pos='n') for token in text]

    stopword = remove_stopword()
    text = [word for word in text if word not in stopword]
    academic_word = remove_academic_word()
    text = [word for word in text if word not in academic_word]
    text = ' '.join(text)

    return text
