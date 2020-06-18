import math
import os
import re

import nltk
import numpy as np

import dbn
import para_reader
import rbm
import sentence
from features import Features


def processFile(file_name):
    # read file from provided folder path
    f = open(file_name, 'r')
    text_0 = f.read()

    # extract content in TEXT tag and remove tags
    text_1 = re.search(r"<TEXT>.*</TEXT>", text_0, re.DOTALL)
    text_1 = re.sub("<TEXT>\n", "", text_1.group(0))
    text_1 = re.sub("\n</TEXT>", "", text_1)

    # replace all types of quotations by normal quotes
    text_1 = re.sub("\n", " ", text_1)

    text_1 = re.sub("\"", "\"", text_1)
    text_1 = re.sub("''", "\"", text_1)
    text_1 = re.sub("``", "\"", text_1)

    text_1 = re.sub(" +", " ", text_1)

    # segment data into a list of sentences
    sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
    lines = sentence_token.tokenize(text_1.strip())

    # setting the stemmer
    sentences = []
    sentencesOriginal = []
    porter = nltk.PorterStemmer()

    # modelling each sentence in file as sentence object
    for line in lines:

        # original words of the sentence before stemming
        originalWords = line[:]
        line = line.strip().lower()

        # word tokenization
        sent = nltk.word_tokenize(line)

        # stemming words
        stemmedSent = [porter.stem(word) for word in sent]
        # stemmedSent = filter(lambda x: x != '.' and x != '`' and x != ',' and x != '?' and x != "'"
        #                                and x != '!' and x != '''"''' and x != "''" and x != "'s", stemmedSent)

        # list of sentence objects
        if stemmedSent != []:
            sentencesOriginal.append(originalWords)
            sentences.append(sentence.sentence(file_name, stemmedSent, originalWords))

    return sentences


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


def TF_IDF(sentences):
    # Method variables
    tfs = TFs(sentences)
    idfs = IDFs(sentences)
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


def sentenceSim(sentence1, sentence2, IDF_w):
    numerator = 0
    denominator = 0

    for word in sentence2.getPreProWords():
        numerator += sentence1.getWordFreq().get(word, 0) * sentence2.getWordFreq().get(word, 0) * IDF_w.get(word,
                                                                                                             0) ** 2

    for word in sentence1.getPreProWords():
        denominator += (sentence1.getWordFreq().get(word, 0) * IDF_w.get(word, 0)) ** 2

    # check for divide by zero cases and return back minimal similarity
    try:
        return numerator / math.sqrt(denominator)
    except ZeroDivisionError:
        return float("-inf")


def buildQuery(sentences, TF_IDF_w, n):
    # sort in descending order of TF-IDF values
    scores = list(TF_IDF_w.keys())
    # scores.sort(reverse=True)
    sorted(scores, reverse=True)
    i = 0
    j = 0
    queryWords = []

    # select top n words
    while (i < n):
        words = TF_IDF_w[scores[j]]
        for word in words:
            queryWords.append(word)
            i = i + 1
            if (i > n):
                break
        j = j + 1

    # return the top selected words as a sentence
    return sentence.sentence("query", queryWords, queryWords)


def bestSentence(sentences, query, IDF):
    best_sentence = None
    maxVal = float("-inf")

    for sent in sentences:
        similarity = sentenceSim(sent, query, IDF)

        if similarity > maxVal:
            best_sentence = sent
            maxVal = similarity
    sentences.remove(best_sentence)

    return best_sentence


def makeSummary(sentences, best_sentence, query, summary_length, lambta, IDF):
    summary = [best_sentence]
    sum_len = len(best_sentence.getPreProWords())

    MMRval = {}

    # keeping adding sentences until number of words exceeds summary length
    while (sum_len < summary_length):
        MMRval = {}

        for sent in sentences:
            MMRval[sent] = MMRScore(sent, query, summary, lambta, IDF)

        maxxer = max(MMRval, key=MMRval.get)
        summary.append(maxxer)
        sentences.remove(maxxer)
        sum_len += len(maxxer.getPreProWords())

    return summary


def MMRScore(Si, query, Sj, lambta, IDF):
    Sim1 = sentenceSim(Si, query, IDF)
    l_expr = lambta * Sim1
    value = [float("-inf")]

    for sent in Sj:
        Sim2 = sentenceSim(Si, sent, IDF)
        value.append(Sim2)

    r_expr = (1 - lambta) * max(value)
    MMR_SCORE = l_expr - r_expr

    return MMR_SCORE


# -------------------------------------------------------------
#	MAIN FUNCTION
# -------------------------------------------------------------
if __name__ == '__main__':

    # set the main Document folder path where the subfolders are present
    main_folder_path = "data/docs"

    # read in all the subfolder names present in the main folder
    for folder in os.listdir(main_folder_path):

        print("Running MMR Summarizer for files in folder: ", folder)
        # for each folder run the MMR summarizer and generate the final summary
        curr_folder = main_folder_path + "/" + folder

        # find all files in the sub folder selected
        files = os.listdir(curr_folder)

        sentencesObject = []
        paragraphs = []
        for file in files:
            sentences = sentences + processFile(curr_folder + "/" + file)


        # calculate TF, IDF and TF-IDF scores
        # TF_w 		= TFs(sentences)
        IDF_w = IDFs(sentences)
        TF_IDF_w = TF_IDF(sentences)

        # build query; set the number of words to include in our query
        query = buildQuery(sentences, TF_IDF_w, 10)

        # pick a sentence that best matches the query
        best1sentence = bestSentence(sentences, query, IDF_w)

        # build summary by adding more relevant sentences
        summary = makeSummary(sentences, best1sentence, query, 100, 0.5, IDF_w)

        final_summary = ""
        for sent in summary:
            final_summary = final_summary + sent.getOriginalWords() + "\n"
        final_summary = final_summary[:-1]
        results_folder = "Results/MMR_results"
        filename=os.path.join(results_folder, (str(folder) + ".MMR"))

        with open(os.path.join(results_folder, (str(folder) + ".MMR")), "w") as fileOut:
            fileOut.write(final_summary)
