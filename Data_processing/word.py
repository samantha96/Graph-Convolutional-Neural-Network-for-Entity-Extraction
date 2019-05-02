import time
import numpy as np
import gensim
import string

from utils import ShowProcess

def word2vec(model_path, in_file_path, out_file_path):

	def load_model(model_path):
		word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
		return word2vec_model

	print("\nLoad Word2Vec model")
	word2vec_model = load_model(model_path)

	print("\nRead sentence from txt")
	content = []
	with open(in_file_path) as f:
	    content = f.readlines()

	print("\nSave vector into file")
	with open(out_file_path, 'a') as out_file:
		process_bar = ShowProcess(len(content))
		for sentence in content:
			words = sentence.split()
			out_file.write("Index " + words[0] + " word vector: \n")
			for word_index in xrange(1,len(words)):
				if words[word_index] in word2vec_model.vocab:
					out_file.write(words[word_index] + ': ')
					for vec in word2vec_model[words[word_index]]:
						out_file.write(' ' + str(vec))
					out_file.write('\n')
				else:
					out_file.write(words[word_index] + ': ' + "N/A" + '\n')
			process_bar.show_process()
	print("\nDone")

if __name__ == '__main__' :

    start = time.time()

    model_path = "/Users/samantha/Desktop/Google_word2vec/GoogleNews-vectors-negative300.bin"
    in_file_path = "/Users/samantha/Desktop/GRAPH/data/CoNLL_sentence_train_slim.txt"
    out_file_path = "/Users/samantha/Desktop/GRAPH/output/CoNLL_word_vec_train.txt"

    word2vec(model_path, in_file_path, out_file_path)

    print ("\nRunning Time: " + str(int(time.time()-start)) + 's')
