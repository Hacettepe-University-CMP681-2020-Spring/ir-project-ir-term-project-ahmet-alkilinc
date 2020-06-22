import collections
import math
import re
import string
import sys

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

import entity2

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')

porter = nltk.PorterStemmer()

stemmer = nltk.stem.porter.PorterStemmer()
WORD = re.compile(r'\w+')
stop = set(stopwords.words('english'))
caps = "([A-Z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"


def similar(tokens_a, tokens_b):
    # Using Jaccard similarity to calculate if two sentences are similar
    ratio = len(set(tokens_a).intersection(tokens_b)) / float(len(set(tokens_a).union(tokens_b)))
    return ratio


def text_to_vector(text):
    words = WORD.findall(text)
    return collections.Counter(words)


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def remove_stop_words(sentences):
    tokenized_sentences = []
    for sentence in sentences:
        tokens = []
        split = sentence.lower().split()
        for word in split:
            if word not in stop:
                try:
                    tokens.append(porter.stem(word))
                except:
                    tokens.append(word)

        tokenized_sentences.append(tokens)
    return tokenized_sentences


def remove_stop_words_without_lower(sentences):
    tokenized_sentences = []
    for sentence in sentences:
        tokens = []
        split = sentence.split()
        for word in split:
            if word.lower() not in stop:
                try:

                    tokens.append(word)
                except:
                    tokens.append(word)

        tokenized_sentences.append(tokens)
    return tokenized_sentences


def split_into_sentences(text):
    text = re.search(r"<TEXT>.*</TEXT>", text, re.DOTALL)
    text = re.sub("<TEXT>\n", "", text.group(0))
    text = re.sub("\n</TEXT>", "", text)

    # text = " " + text + "  "
    # text = text.replace("\n", " ")
    # text = re.sub(prefixes, "\\1<prd>", text)
    # text = re.sub(websites, "<prd>\\1", text)
    # if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    # text = re.sub("\s" + caps + "[.] ", " \\1<prd> ", text)
    # text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    # text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    # text = re.sub(caps + "[.]" + caps + "[.]", "\\1<prd>\\2<prd>", text)
    # text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    # text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    # text = re.sub(" " + caps + "[.]", " \\1<prd>", text)
    # if "”" in text: text = text.replace(".”", "”.")
    # if "\"" in text: text = text.replace(".\"", "\".")
    # if "!" in text: text = text.replace("!\"", "\"!")
    # if "?" in text: text = text.replace("?\"", "\"?")
    #
    # text = text.replace(".", ".<stop>")
    # text = text.replace("?", "?<stop>")
    # text = text.replace("!", "!<stop>")

    # replace all types of quotations by normal quotes
    text = re.sub("\n", " ", text)

    text = re.sub("\"", " ", text)
    text = re.sub("\'", " ", text)
    text = re.sub("''", " ", text)
    text = re.sub("``", " ", text)

    text = re.sub(" +", " ", text)

    text = text.translate(string.punctuation)

    # text = text.replace("<prd>", ".")
    # sentences = text.split("<stop>")
    # segment data into a list of sentences
    sentences = nltk.sent_tokenize(text)
    # sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def posTagger(tokenized_sentences):
    tagged = []
    for sentence in tokenized_sentences:
        tag = nltk.pos_tag(sentence)
        tagged.append(tag)
    return tagged


def TFs(sentences):
    # initialize tfs dictonary
    tfs = {}

    # for every sentence in document cluster
    for sent in sentences:
        # retrieve word frequencies from sentence object
        wordFreqs = sent.getWordFreq()

        # for every word
        for word in wordFreqs.keys():
            # if word already present in the dictonary
            if tfs.get(word, 0) != 0:
                tfs[word] = tfs[word] + wordFreqs[word]
            # else if word is being added for the first time
            else:
                tfs[word] = wordFreqs[word]
    return tfs


def IDFs(sentences):
    N = len(sentences)
    idf = 0
    idfs = {}
    words = {}
    w2 = []

    # every sentence in our cluster
    for sent in sentences:

        # every word in a sentence
        for word in sent.getPreProWords():

            # not to calculate a word's IDF value more than once
            if sent.getWordFreq().get(word, 0) != 0:
                words[word] = words.get(word, 0) + 1

    # for each word in words
    for word in words:
        n = words[word]

        # avoid zero division errors
        try:
            w2.append(n)
            idf = np.math.log10(float(N) / n)
        except ZeroDivisionError:
            idf = 0

        # reset variables
        idfs[word] = idf

    return idfs


def tfIsf(tokenized_sentences):
    scores = []
    COUNTS = []
    for sentence in tokenized_sentences:
        counts = collections.Counter(sentence)
        isf = []
        score = 0
        for word in counts.keys():
            count_word = 1
            for sen in tokenized_sentences:
                for w in sen:
                    if word == w:
                        count_word += 1
            score = score + counts[word] * math.log(count_word - 1)
        if len(sentence) != 0:
            scores.append(score / len(sentence))
        else:
            scores.append(0.0)
    return scores


class Features(object):

    def __init__(self, sentences, sentences_object, paragraphs):
        self.paragraphs = paragraphs
        self.sentences = sentences
        self.sentencesObject = sentences_object
        self.tokenized_sentences = remove_stop_words(self.sentences)
        self.tagged = posTagger(self.tokenized_sentences)
        self.tfs = TFs(sentences_object)
        self.idf = IDFs(sentences_object)
        self.pos_Taggers = posTagger(self.tokenized_sentences)
        self.similarity_Scores = self.similarityScores()
        self.proper_NounScores = self.properNounScores()
        self.tfIsf = tfIsf(self.tokenized_sentences)
        self.centroid_Similarity = self.centroidSimilarity()
        self.numeric_Token = self.numericToken()
        self.named_EntityRecog = self.namedEntityRecog()
        self.sentence_Pos = self.sentencePos()
        self.sentence_Length = self.sentenceLength()
        self.thematic_Feature = self.thematicFeature()
        self.upperCase_Feature = self.upperCaseFeature()
        self.sentence_Position = self.sentencePosition()
        self.numWords = self.num_words()
        self.numUniqWords = self.num_uniq_words()
        self.numChars = self.num_chars()
        self.numWordsUpper = self.num_words_upper()
        self.numWordsTitle = self.num_words_title()
        self.meanWordLen = self.mean_word_len()
        self.numCharacterLen = self.num_character_len()

    def getParagraphs(self):
        return self.paragraphs

    def getSentences(self):
        return self.sentences

    def getTokenizedSentences(self):
        return self.tokenized_sentences

    def getTagged(self):
        return self.tagged

    def getTfs(self):
        return self.tfs

    def getIdf(self):
        return self.idf

    def getPosTaggers(self):
        return self.pos_Taggers

    def getSimilarityScores(self):
        return self.similarity_Scores

    def getProperNounScores(self):
        return self.proper_NounScores

    def getTfIsf(self):
        return self.tfIsf

    def getCentroidSimilarity(self):
        return self.centroid_Similarity

    def getNumericToken(self):
        return self.numeric_Token

    def getNamedEntityRecog(self):
        return self.named_EntityRecog

    def getSentencePos(self):
        return self.sentence_Pos

    def getSentenceLength(self):
        return self.sentence_Length

    def getThematicFeature(self):
        return self.thematic_Feature

    def getUpperCaseFeature(self):
        return self.upperCase_Feature

    def getSentencePosition(self):
        return self.sentence_Position

    def getNumWords(self):
        return self.numWords

    def getNumUniqWords(self):
        return self.numUniqWords

    def getNumChars(self):
        return self.numChars

    def getNumWordsUpper(self):
        return self.numWordsUpper

    def getNumWordsTitle(self):
        return self.numWordsTitle

    def getMeanWordLen(self):
        return self.meanWordLen

    def getNumCharacterLen(self):
        return self.numCharacterLen

    def num_words(self):
        num_words = []
        data = []

        for sentence in self.sentences:
            data.append(sentence)

        df = pd.DataFrame(data, columns=['text'])
        df['num_words'] = df['text'].apply(lambda x: len(str(x).split()))
        for i in range(df.shape[0]):
            num_words.append(df['num_words'].iloc[i])

        return num_words

    def num_uniq_words(self):
        num_uniq_words = []
        data = []

        for sentence in self.sentences:
            data.append(sentence)

        df = pd.DataFrame(data, columns=['text'])
        df['num_uniq_words'] = df['text'].apply(lambda x: len(set(str(x).split())))
        for i in range(df.shape[0]):
            num_uniq_words.append(df['num_uniq_words'].iloc[i])

        return num_uniq_words

    def num_chars(self):
        num_chars = []
        data = []

        for sentence in self.sentences:
            data.append(sentence)

        df = pd.DataFrame(data, columns=['text'])
        df['num_chars'] = df['text'].apply(lambda x: len(str(x)))
        for i in range(df.shape[0]):
            num_chars.append(df['num_chars'].iloc[i])

        return num_chars

    def num_words_upper(self):
        num_words_upper = []
        data = []

        for sentence in self.sentences:
            data.append(sentence)

        df = pd.DataFrame(data, columns=['text'])
        df['num_words_upper'] = df['text'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
        for i in range(df.shape[0]):
            num_words_upper.append(df['num_words_upper'].iloc[i])

        return num_words_upper

    def num_words_title(self):
        num_words_title = []
        data = []

        for sentence in self.sentences:
            data.append(sentence)

        df = pd.DataFrame(data, columns=['text'])
        df['num_words_title'] = df['text'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
        for i in range(df.shape[0]):
            num_words_title.append(df['num_words_title'].iloc[i])

        return num_words_title

    def mean_word_len(self):
        mean_word_len = []
        data = []

        for sentence in self.sentences:
            data.append(sentence)

        df = pd.DataFrame(data, columns=['text'])
        df['mean_word_len'] = df['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
        for i in range(df.shape[0]):
            mean_word_len.append(df['mean_word_len'].iloc[i])

        return mean_word_len

    def num_character_len(self):
        num_character_len = []
        data = []

        for sentence in self.sentences:
            data.append(sentence)

        df = pd.DataFrame(data, columns=['text'])
        df['num_character_len'] = df['text'].apply(lambda x: len(x))
        for i in range(df.shape[0]):
            num_character_len.append(df['num_character_len'].iloc[i])

        return num_character_len

    def TF_IDF(self):
        # Method variables
        tfs = TFs(self.sentences)
        idfs = IDFs(self.sentences)
        retval = {}

        # for every word
        for word in tfs:
            # calculate every word's tf-idf score
            tf_idfs = tfs[word] * idfs[word]

            # add word and its tf-idf score to dictionary
            if retval.get(tf_idfs, None) == None:
                retval[tf_idfs] = [word]
            else:
                retval[tf_idfs].append(word)

        return retval

    def similarityScores(self):
        scores = []
        for sentence in self.tokenized_sentences:
            score = 0;
            for sen in self.tokenized_sentences:
                if sen != sentence:
                    score += similar(sentence, sen)
            scores.append(score)
        return scores

    def properNounScores(self):
        scores = []
        for i in range(len(self.tagged)):
            score = 0
            for j in range(len(self.tagged[i])):
                if self.tagged[i][j][1] == 'NNP' or self.tagged[i][j][1] == 'NNPS':
                    score += 1
            if len(self.tagged[i]) != 0:
                scores.append(score / float(len(self.tagged[i])))
            else:
                scores.append(0.0)
        return scores

    def centroidSimilarity(self):
        centroidIndex = self.tfIsf.index(max(self.tfIsf))
        scores = []
        for sentence in self.sentences:
            vec1 = text_to_vector(self.sentences[centroidIndex])
            vec2 = text_to_vector(sentence)

            score = get_cosine(vec1, vec2)
            scores.append(score)
        return scores

    def numericToken(self):
        scores = []
        for sentence in self.tokenized_sentences:
            score = 0
            for word in sentence:
                if is_number(word):
                    score += 1

            if len(sentence) != 0:
                scores.append(score / float(len(sentence)))
            else:
                scores.append(0.0)

        return scores

    def namedEntityRecog(self):
        counts = []
        for sentence in self.sentences:
            count = entity2.ner(sentence)
            counts.append(count)
        return counts

    def sentencePos(self):
        th = 0.2
        minv = th * len(self.sentences)
        maxv = th * 2 * len(self.sentences)
        pos = []
        for i in range(len(self.sentences)):
            if i == 0 or i == len((self.sentences)):
                pos.append(0)
            else:
                t = math.cos((i - minv) * ((1 / maxv) - minv))
                pos.append(t)

        return pos

    def sentenceLength(self):
        count = []
        maxLength = sys.maxsize
        for sentence in self.tokenized_sentences:
            num_words = 0
            for word in sentence:
                num_words += 1
            if num_words < 3:
                count.append(0)
            else:
                count.append(num_words)

        count = [1.0 * x / (maxLength) for x in count]
        return count

    def thematicFeature(self):
        word_list = []
        for sentence in self.tokenized_sentences:
            for word in sentence:
                try:
                    word = ''.join(e for e in word if e.isalnum())
                    # print(word)
                    word_list.append(word)
                except Exception as e:
                    print("ERR")
        counts = collections.Counter(word_list)
        number_of_words = len(counts)
        most_common = counts.most_common(10)
        thematic_words = []
        for data in most_common:
            thematic_words.append(data[0])
        # print(thematic_words)
        scores = []
        for sentence in self.tokenized_sentences:
            score = 0
            for word in sentence:
                try:
                    word = ''.join(e for e in word if e.isalnum())
                    if word in thematic_words:
                        score = score + 1
                    # print(word)
                except Exception as e:
                    print("ERR")
            score = 1.0 * score / (number_of_words)
            scores.append(score)
        return scores

    def upperCaseFeature(self):
        tokenized_sentences2 = remove_stop_words_without_lower(self.sentences)
        # print(tokenized_sentences2)
        upper_case = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        scores = []
        for sentence in tokenized_sentences2:
            score = 0
            for word in sentence:
                if word[0] in upper_case:
                    score = score + 1

            if len(sentence) != 0:
                scores.append(1.0 * score / len(sentence))
            else:
                scores.append(0.0)

        return scores

    def sentencePosition(self):
        scores = []
        for para in self.paragraphs:
            sentences = split_into_sentences(para)
            # print(len(sentences))
            if len(sentences) == 1:
                scores.append(1.0)
            elif len(sentences) == 2:
                scores.append(1.0)
                scores.append(1.0)
            else:
                scores.append(1.0)
                for x in range(len(sentences) - 2):
                    scores.append(0.0)
                scores.append(1.0)
        return scores
