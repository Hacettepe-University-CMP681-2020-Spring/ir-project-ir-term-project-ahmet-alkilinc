import nltk
import re
import string


class Preprocess:

    def __init__(self, sentence):
        self.sentence = sentence
        print("Preprocessing...")

    def replace_nonAlphabetic(sentence):
        sentence = [re.sub(r'[^a-z]', ' ', text.lower()) for text in sentence]

        return sentence

    def replace_quotations(sentence):
        sentence = re.sub("\n", " ", sentence)

        sentence = re.sub("\"", "\"", sentence)
        sentence = re.sub("''", "\"", sentence)
        sentence = re.sub("``", "\"", sentence)
        sentence = re.sub(" +", " ", sentence)

        return sentence

    def remove_puctiaation(sentence):
        sentence = sentence.translate(string.punctuation)

        return sentence

    def remove_stopword(sentence):
        stops = nltk.corpus.stopwords.words('english')
        sentence = [[word for word in text if word not in stops] for text in sentence]

        return sentence

    def stemming(sentence):
        porter = nltk.stem.porter.PorterStemmer()
        sentence = [[porter.stem(word) for word in text] for text in sentence]

        return sentence
