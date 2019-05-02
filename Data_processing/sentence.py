import time
import numpy as np
import string

from nltk.parse.stanford import StanfordDependencyParser
from utils import ShowProcess
from string import punctuation

def dependency_parser(parser_path, model_path, in_file_path, out_file_path):

	def load_model(parser_path, model_path):
		dependencyParser = StanfordDependencyParser(path_to_jar=parser_path, path_to_models_jar=model_path)
		return dependencyParser

	print("\nLoad dependency parser")
	dependencyParser = load_model(parser_path, model_path)

	print("\nRead sentence from txt")
	content = []
	with open(in_file_path) as f:
	    content = f.readlines()

	print("\nSave word pair into file")
	with open(out_file_path, 'a') as out_file:
		process_bar = ShowProcess(len(content))
		for sentence in content:
			index, space, rest_sentence = sentence.partition(' ')
			rest_sentence = ''.join(i for i in rest_sentence if not i.isdigit())
			rest_sentence = ''.join(c for c in rest_sentence if c not in punctuation)

			if rest_sentence.count(' ') > 1 and 2 * rest_sentence.count(' ') < len(rest_sentence):
				result = dependencyParser.raw_parse(rest_sentence)
				parsetree = list(result)[0]

				out_file.write("Index " + index.rstrip("\n\r") + " word pairs:\n")
				if parsetree.nodes.values() > 1:
					word_list = list()
					count = 1
					for k in parsetree.nodes.values():
						if k["head"] is not None:
							if str(k["address"]) != str(count):
								word_list.append("xxxxxxxx")
								count = count + 1
							word_list.append(k["word"])
							count = count + 1
					for k in parsetree.nodes.values():
						if k["head"] is not None and len(list(k["deps"])) != 0:
							for elem in k["deps"].values():
								for each in elem:
									# print (each - 1)
									out_file.write(' (' + k["word"] + '-' + str(k["address"]) + ',' + word_list[each - 1] + '-' + str(each) + ')')
				out_file.write('\n\n')
			else:
				out_file.write("Index " + index.rstrip("\n\r") + " word pairs: \n")
				out_file.write(' ')
				out_file.write('\n\n')
			process_bar.show_process()
	print("\nDone")

if __name__ == '__main__' :

    start = time.time()

    parser_path = "/Users/samantha/Desktop/stanford/stanford-parser.jar"
    model_path = "/Users/samantha/Desktop/stanford/stanford-parser-3.7.0-models.jar"
    in_file_path = "/Users/samantha/Desktop/GRAPH/data/CoNLL_sentence_train.txt"
    out_file_path = "/Users/samantha/Desktop/GRAPH/output/sentence_train_dependency.txt"

    dependency_parser(parser_path, model_path, in_file_path, out_file_path)

    print ("\nRunning Time: " + str(int(time.time()-start)) + 's')
