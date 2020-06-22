import nltk
import os
import string
import re

from lexrank import STOPWORDS, LexRank
from path import Path


def split_into_sentences(text):
    text = re.search(r"<TEXT>.*</TEXT>", text, re.DOTALL)
    text = re.sub("<TEXT>\n", "", text.group(0))
    text = re.sub("\n</TEXT>", "", text)

    text = re.sub("\n", " ", text)

    text = re.sub("\"", "\"", text)
    text = re.sub("''", "\"", text)
    text = re.sub("``", "\"", text)
    text = re.sub(" +", " ", text)

    text = text.translate(string.punctuation)

    # text = text.replace("<prd>", ".")
    # sentences = text.split("<stop>")
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

        print("Running Lexrank Summarizer for files in folder: ", folder)
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
            file = open(curr_folder + "/" + file, 'r')
            text = file.read()
            sentences = sentences + split_into_sentences(text)

        extracted_sentences = lxr.get_summary(sentences, summary_size=10, threshold=.1)
        print(extracted_sentences)

        scores_cont = lxr.rank_sentences(
            sentences,
            threshold=None,
            fast_power_method=False, )

        # print(scores_cont)

        final_summary = ""
        print("\n\n\nExtracted Final Text : \n\n\n")
        for i in range(len(extracted_sentences)):
            # print("\n" + extracted_sentences[i][0])
            if len(final_summary.split()) <= 100:
                if len(extracted_sentences[i].split()) >= 5:
                    final_summary = final_summary + extracted_sentences[i] + "\n"
            else:
                break

        results_folder = "Results/LEXRANK_results/"
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        filename = os.path.join(results_folder, (str(folder) + ".LEXRANK"))
        with open(os.path.join(results_folder, (str(folder) + ".LEXRANK")), "w") as fileOut:
            fileOut.write(final_summary)
