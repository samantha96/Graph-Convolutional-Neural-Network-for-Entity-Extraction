# This program mainly do the preprocess of GCN Entity Extraction project.
# Provide functions that generate three matrix for GCN's augments

#-----------------------------------------------------------------
# Improve functions, for sentences, dependencies, vectors and classes,
# use a dictionary to store them, only need to set the index of sentence.
# Run this program to store four  dictionary in the local memory.
# Extract needed matrix using find functions, or use operate[]
#-----------------------------------------------------------------


import time
from numpy import *
import numpy as np
import re
import scipy.sparse as sp

# get all sentences from a file
# set the first elem as key and others are set as value
def getSentences(dataset_path):
    sentences = {}
    indexes = []
    for line in open(dataset_path):
        line = line.split()
        index = line[0]
        sentence = line[1:]
        sentences[index] = sentence
        indexes.append(index)
    return sentences, indexes


# select sentences, return sentences that
# word number is greater than number
def select(sentences, number):
    print("Reading the sentences ...\n")
    sens = {}
    for sen in sentences.items():
        if (len(sen[1]) >= number):
            sens[sen[0]] = sen[1]
    return sens

# get dependencies from a file
# set the index as key and dependency matrix as value
# can get the dependency matrix by the given index
def getDependencies(dependency_path, sentences):
    print("Reading the dependencies...\n")
    deps = {}
    temp = []
    for line in open(dependency_path):
        line = line.split()
        temp.append(line)
    for i in range(len(temp)):
        if(temp[i] != [] and temp[i][0] == "Index"):
            index = temp[i][1]
            length = len(sentences[index])
            matrix = zeros([length, length], dtype=int8)
            for dep in temp[i+1]:
                dep = re.split('-|,|\(|\)', dep)
                x = int(dep[2]) - 1
                y = int(dep[4]) - 1
                matrix[x][y] = 1
            deps[index] = matrix
    return deps


def findMatrix(index, dependencies):
    return dependencies[index]


# get word vectors from a file
# use index as key and dependency matrix as value
def getVectors(wordvector_path, sentences):
    print("Reading the vectors...\n")
    vecs = {}
    temp = []
    for line in open(wordvector_path):
        line = line.split()
        temp.append(line)
    for i in range(len(temp)):
        print(i)
        if(temp[i][0] == "Index"):
            index = temp[i][1]
            if(index not in sentences):
                continue
            length = len(sentences[index])
            matrix = zeros([length, 300], dtype=float)
            for j in range(length):
                vector = temp[i+j+1]
                if(vector[1] == "N/A"):
                    continue
                else:
                    for k in range(300):
                        matrix[j][k] = float(vector[k+1])
            vecs[index] = matrix
    return vecs


def findVectors(index, vectors):
    return vectors[index]

# get word label from a file
# use index as key and label matrix as value
def getClass(wordclass_path, sentences):
    print("Reading the classes...\n")
    classes = {}
    temp = []
    for line in open(wordclass_path):
        line = line.replace("<", "|<")
        line = line.replace(">", "> ")
        line = re.split("\|", line)
        index = line[0]
        line = line[1:]
        if (index not in sentences):
            continue
        matrix = zeros([5, len(sentences[index])], dtype=int8)
        for item in line:
            item = item.split()
            sentence = np.array(sentences[index])
            if(item[0] == "<O>"):
                objects = item[1:]
                for obj in objects:
                    ii = np.where(sentence == obj)[0]
                    for i in ii:
                        matrix[0][i] = 1
            if(item[0] == "<ORG>"):
                orgs = item[1:]
                for org in orgs:
                    ii = np.where(sentence == org)[0]
                    for i in ii:
                        matrix[1][i] = 1
            if(item[0] == "<PER>"):
                persons = item[1:]
                for person in persons:
                    ii = np.where(sentence == person)[0]
                    for i in ii:
                        matrix[2][i] = 1
            if (item[0] == "<MISC>"):
                maliciouses = item[1:]
                for malicious in maliciouses:
                    ii = np.where(sentence == malicious)[0]
                    for i in ii:
                        matrix[3][i] = 1
            if (item[0] == "<LOC>"):
                locations = item[1:]
                for location in locations:
                    ii = np.where(sentence == location)[0]
                    for i in ii:
                        matrix[4][i] = 1
        for n in range(len(sentences[index])):
            isNone = 1
            for m in range(5):
                if(matrix[m][n] == 1):
                    isNone = 0
            if(isNone == 1):
                matrix[0][n] = 1
        classes[index] = matrix

    return classes


def findClasses(index, classes):
    return classes[index]

def getMatrix(dependencies, vectors, classes, index):
    temp = []
    matrixNN = dependencies[index]
    for i in range(len(matrixNN)):
        for j in range(len(matrixNN[0])):
            if(matrixNN[i][j] == 1):
                temp.append((i, j))
    temp = np.array(temp)
    adj = sp.coo_matrix((np.ones(temp.shape[0]), (temp[:, 0], temp[:, 1])),
                        shape=(matrixNN.shape[0], matrixNN.shape[0]), dtype=np.float32)
    matrixND = vectors[index]
    matrixNE = classes[index]
    matrixNE = np.transpose(matrixNE)
    return matrixND, adj, matrixNE

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def getSplit(y):
    idx_train = range((y.shape[0]) / 2)
    idx_val = range((y.shape[0])/2, y.shape[0])
    idx_test = range((y.shape[0])/2, y.shape[0])
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask



if __name__ == '__main__' :
    start = time.time()
    sentence_set = "/Users/samantha/Desktop/Graph/data/sentence.txt"
    dependencies_set = "/Users/samantha/Desktop//Graph/output/sentence_dependency.txt"
    wordvector_set = "/Users/samantha/Desktop/Graph/output/word_vec.txt"
    wordclass_set = "/Users/samantha/Desktop/Graph/output/word_class.txt"
    sentences, indexes = getSentences(sentence_set)
    dependencies = getDependencies(dependencies_set, sentences)
    vectors = getVectors(wordvector_set, sentences)
    classes = getClass(wordclass_set, sentences)
    MND, temp, MNE = getMatrix(dependencies, vectors, classes, indexes[19])
    y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = getSplit(MNE)

    print("pause")



