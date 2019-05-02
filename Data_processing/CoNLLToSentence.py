
import time
from numpy import *
from copy import deepcopy

punc = [',', '.', '\"', '-', '-DOCSTART-', '\'', '(', ')', ':', '...', '/', '=', '--'];


def generateList(input_path):
    list = []
    for line in open (input_path):
        line = line.split()
        list.append(line)
    return list

def generateSentenceFile(output_path, list):
    index = 1
    sentences = []
    sentence = ""
    for i in range(len(list)):
        if(list[i] == []):
            sentences.append(sentence)
            sentence = ""
        else:
            if (punc.__contains__(list[i][0])):
                continue
            sentence += list[i][0]
            sentence += " "

    with open (output_path, "w") as file:
        for sen in sentences:
            if(sen == ""):
                continue
            file.write(str(index))
            file.write(" ")
            index += 1
            file.write(sen)
            file.write('\r\n')


def generateClassFile(output_path, list):
    index = 1
    wordClass = ["<O> ", "<ORG> ", "<PER> ", "<MISC> ", "<LOC> "]
    currentClass = deepcopy(wordClass)
    wordClasses = []
    for i in range(len(list)):
        if(list[i] == []):
            wordClasses.append(currentClass)
            currentClass = deepcopy(wordClass)
        else:
            if(punc.__contains__(list[i][0])):
                continue
            if(list[i][3] == "O"):
                currentClass[0] += list[i][0]
                currentClass[0] += " "
            if(list[i][3] == "B-ORG" or list[i][3] == "I-ORG"):
                currentClass[1] += list[i][0]
                currentClass[1] += " "
            if (list[i][3] == "B-PER" or list[i][3] == "I-PER"):
                currentClass[2] += list[i][0]
                currentClass[2] += " "
            if (list[i][3] == "B-MISC" or list[i][3] == "I-MISC"):
                currentClass[3] += list[i][0]
                currentClass[3] += " "
            if (list[i][3] == "B-LOC" or list[i][3] == "I-LOC"):
                currentClass[4] += list[i][0]
                currentClass[4] += " "

    with open (output_path, "w") as file:
        for cla in wordClasses:
            if(cla == wordClass):
                continue
            file.write(str(index))
            index += 1
            file.write('|')
            for label in cla:
                file.write(label)
            file.write('\r\n')
if __name__ == '__main__' :
    start = time.time()
    input_trainingset = "/Users/samantha/Desktop/Graph/data/CoNLL_train.txt"
    input_testingset = "/Users/samantha/Desktop/Graph/data/CoNLL_test.txt"
    output_trainingset = "/Users/samantha/Desktop/Graph/data/CoNLL_sentence_train.txt"
    output_testingset = "/Users/samantha/Desktop/Graph/data/CoNLL_sentence_test.txt"
    output_class_train = "/Users/samantha/Desktop/Graph/data/CoNLL_class_train.txt"
    output_class_test = "/Users/samantha/Desktop/Graph/data/CoNLL_class_test.txt"
    output_class_train_index = "/Users/samantha/Desktop/Graph/data/CoNLL_class_train_index.txt"
    output_class_test_index = "/Users/samantha/Desktop/Graph/data/CoNLL_class_test_index.txt"
    list = generateList(input_trainingset)
    # list = generateList(input_testingset)
    # generateSentenceFile(output_trainingset, list)
    # generateSentenceFile(output_testingset, list)
    # generateClassFile(output_class_train, list)
    # generateClassFile(output_class_test, list)
    generateClassFile(output_class_train_index, list)
    print("pause")
