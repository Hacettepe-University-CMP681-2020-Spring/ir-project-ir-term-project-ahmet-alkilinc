
import nltk
import numpy as np
import pandas as pd
import re


class Featuresz"":

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
                idf = math.log10(float(N) / n)
            except ZeroDivisionError:
                idf = 0

            # reset variables
            idfs[word] = idf

        return idfs