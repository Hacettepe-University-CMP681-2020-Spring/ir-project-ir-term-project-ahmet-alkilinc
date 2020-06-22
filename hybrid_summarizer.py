import nltk
import os
import string
import re
import numpy as np
from sklearn import preprocessing

import rbm

import para_reader


from features import Features
from mmr_summarizer import processFile

from lexrank import STOPWORDS, LexRank
from path import Path


def split_into_sentences(text):
    text = re.search(r"<TEXT>.*</TEXT>", text, re.DOTALL)
    text = re.sub("<TEXT>\n", "", text.group(0))
    text = re.sub("\n</TEXT>", "", text)

    # replace all types of quotations by normal quotes
    text = re.sub("\n", " ", text)

    text = re.sub("\"", "\"", text)
    text = re.sub("''", "\"", text)
    text = re.sub("``", "\"", text)
    text = re.sub(" +", " ", text)

    text = text.translate(string.punctuation)


    # segment data into a list of sentences
    sentences = nltk.sent_tokenize(text)
    # sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


if __name__ == '__main__':

    # set the main Document folder path where the subfolders are present
    main_folder_path = "data/docs"

    # read in all the subfolder names present in the main folder
    for folder in os.listdir(main_folder_path):

        print("Running HYBRID Summarizer for files in folder: ", folder)
        # for each folder run the MMR summarizer and generate the final summary
        curr_folder = main_folder_path + "/" + folder

        # find all files in the sub folder selected
        files = os.listdir(curr_folder)

        sentencesObject = []
        paragraphs = []
        sentences = []

        documents = []
        documents_dir = Path(curr_folder)

        for file_path in documents_dir.files('*'):
            with file_path.open(mode='rt', encoding='utf-8') as fp:
                documents.append(fp.readlines())

        lxr = LexRank(documents, stopwords=STOPWORDS['en'])


        for file in files:
            sentencesObject = sentencesObject + processFile(curr_folder + "/" + file)
            paragraphs = paragraphs + para_reader.show_paragraphs(curr_folder + "/" + file)

            file = open(curr_folder + "/" + file, 'r')
            text = file.read()
            sentences = sentences + split_into_sentences(text)

        lexrank_scores = lxr.rank_sentences(
            sentences,
            threshold=None,
            fast_power_method=False, )

        print(lexrank_scores)
        print(len(lexrank_scores))
        print(len(sentences))

        features = Features(sentences, sentencesObject, paragraphs)
        # sentences = sentence+features.getSentences()
        tokenized_sentences = features.getTokenizedSentences()

        tfIsfScore = features.getTfIsf()
        similarityScore = features.getSimilarityScores()

        properNounScore = features.getProperNounScores()
        centroidSimilarityScore = features.getCentroidSimilarity()
        numericTokenScore = features.getNumericToken()
        namedEntityRecogScore = features.getNamedEntityRecog()
        sentencePosScore = features.getSentencePos()
        sentenceLengthScore = features.getSentenceLength()
        thematicFeatureScore = features.getThematicFeature()
        sentenceParaScore = features.getSentencePosition()
        numWords = features.getNumWords()
        numUniqWords = features.getNumUniqWords()
        numChars = features.getNumChars()
        numWordsUpper = features.getNumWordsUpper()
        numWordsTitle = features.getNumWordsTitle()
        meanWordLen = features.getMeanWordLen()
        numCharacterLen = features.getNumCharacterLen()

        featureMatrix = [thematicFeatureScore, sentencePosScore, sentenceLengthScore, properNounScore,
                         numericTokenScore, namedEntityRecogScore, tfIsfScore, centroidSimilarityScore]

        # featureMatrix.append(sentenceParaScore)

        featureMat = np.zeros((len(features.getSentences()), 8))
        for i in range(8):
            for j in range(len(features.getSentences())):
                featureMat[j][i] = featureMatrix[i][j]

        # print("\n\n\nPrinting Feature Matrix : ")
        # print(featureMat)
        # print("\n\n\nPrinting Feature Matrix Normed : ")
        # featureMat_normed = featureMat / featureMat.max(axis=0)

        featureMat_normed = preprocessing.normalize(featureMat)
        # featureMat_normed = featureMat

        feature_sum = []

        for i in range(len(np.sum(featureMat, axis=1))):
            feature_sum.append(np.sum(featureMat, axis=1)[i])

        # print(featureMat_normed)
        # for i in range(len(features.getSentences())):
        #     print(featureMat_normed[i])



        temp = rbm.test_rbm(dataset=featureMat_normed, learning_rate=0.1, training_epochs=14, batch_size=5, n_chains=5,
                            n_hidden=8)

        # temp = dbn.test_DBN(dataset=featureMat, finetune_lr=0.1, pretraining_epochs=100,
        #              pretrain_lr=0.01, k=1, training_epochs=1000,
        #              batch_size=10)

        # print("\n\n")
        # print("np.sum(temp, axis=1)")
        # print(np.sum(temp, axis=1))
        enhanced_feature_sum = []
        enhanced_feature_sum2 = []

        for i in range(len(np.sum(temp, axis=1))):
            enhanced_feature_sum.append([(np.sum(temp, axis=1)[i] + lexrank_scores[i])/2, i])
            enhanced_feature_sum2.append(np.sum(temp, axis=1)[i])

        # print(enhanced_feature_sum)
        # print("\n\n\n")

        enhanced_feature_sum.sort(key=lambda x: x[0])
        # print(enhanced_feature_sum)

        length_to_be_extracted = int(len(enhanced_feature_sum) / 2)

        # print("\n\nThe text is : \n\n")
        # for x in range(len(features.getSentences())):
        #     print(sentences[x])

        # print("\n\n\nExtracted sentences : \n\n\n")
        extracted_sentences = []
        extracted_sentences.append([sentences[0], 0])

        indeces_extracted = []
        indeces_extracted.append(0)

        for x in range(length_to_be_extracted):
            if (enhanced_feature_sum[x][1] != 0):
                extracted_sentences.append([sentences[enhanced_feature_sum[x][1]], enhanced_feature_sum[x][1]])
                indeces_extracted.append(enhanced_feature_sum[x][1])

        extracted_sentences.sort(key=lambda x: x[1])

        final_summary = ""
        print("\n\n\nExtracted Final Text : \n\n\n")
        for i in range(len(extracted_sentences)):
            # print("\n" + extracted_sentences[i][0])
            if len(final_summary.split()) <= 100:
                if len(extracted_sentences[i][0].split()) >= 5:
                    final_summary = final_summary + extracted_sentences[i][0] + "\n"
            else:
                break

        results_folder = "Results/HYBRID_results"
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        filename = os.path.join(results_folder, (str(folder) + ".HYBRID"))
        with open(os.path.join(results_folder, (str(folder) + ".HYBRID")), "w") as fileOut:
            fileOut.write(final_summary)
