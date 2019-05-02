from preprocess import *

sentence_set = "/Users/samantha/Desktop/Graph/data/sentence.txt"
dependencies_set = "/Users/samantha/Desktop/Graph/output/sentence_dependency.txt"
wordvector_set = "/Users/samantha/Desktop/Graph/output/word_vec.txt"
wordclass_set = "/Users/samantha/Desktop/Graph/output/word_class.txt"

sentences, indexes = getSentences(sentence_set)
dependencies = getDependencies(dependencies_set, sentences)
vectors = getVectors(wordvector_set, sentences)
classes = getClass(wordclass_set, sentences)
MND, MNN, MNE = getMatrix(dependencies, vectors, classes, indexes[0])
for i in range(1, 10):
    key = indexes[i]
    MNDi, MNNi, MNEi = getMatrix(dependencies, vectors, classes, indexes[i])
    MND = vstack((MND, MNDi))
    MNE = vstack((MNE, MNEi))
    right_top = zeros((MNN.shape[0], MNNi.shape[1]), dtype=int8)
    left_bottom = zeros((MNNi.shape[0], MNN.shape[1]), dtype = int8)
    left = vstack((MNN, left_bottom))
    right = vstack((right_top, MNNi))
    MNN = hstack((left, right))
adj = toTuple(MNN)
y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = getSplit(MNE)










print("pause")
