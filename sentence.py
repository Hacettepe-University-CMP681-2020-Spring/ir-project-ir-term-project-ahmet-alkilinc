

class sentence(object):

    def __init__(self, docName, preproWords, originalWords):
        self.docName = docName
        self.preproWords = preproWords
        self.wordFrequencies = self.sentenceWordFreq()
        self.originalWords = originalWords

    def getDocName(self):
        return self.docName

    def getWordFreq(self):
        return self.wordFrequencies

    def sentenceWordFreq(self):
        wordFreq = {}
        for word in self.preproWords:
            if word not in wordFreq.keys():
                wordFreq[word] = 1
            else:
                # if word in stopwords.words('english'):
                # 	wordFreq[word] = 1
                # else:
                wordFreq[word] = wordFreq[word] + 1
        return wordFreq