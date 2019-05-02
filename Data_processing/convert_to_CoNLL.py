import time
import numpy as np
import string
import nltk
import re

from nltk.parse.stanford import StanfordDependencyParser
from utils import ShowProcess

def getClass(wordclass_path):
    classes = {}
    index = 0
    for line in open(wordclass_path):
        line = line.replace("<", "|<")
        line = line.replace(">", "> ")
        line = re.split("\|", line)
        line = line[1:]
        classes[index] = line
        index = index + 1
    return classes


def get_label(word, classes, count):
	label = "Other"
	ind = 0
	for item in classes[count]:
		item = item.split()
		ind = ind + 1
		# any(str(word) in s for s in list(item[1:]))
		if word in list(item[1:]):
			if item[0] == "<Agent>":
				label = "Agent"
			elif item[0] == "<verb>":
				label = "Verb"
			elif item[0] == "<Theme>" or item[0] == "<Place>" or item[0] == "<Location>":
				label = "Location"
	return label

def dependency_parser(in_file_path, wordclass_set, out_file_path):
	print("\nRead sentence from txt")
	content = []
	with open(in_file_path) as f:
	    content = f.readlines()

	print ("\nLoad word classes")
	classes = []
	classes = getClass(wordclass_set)

	print("\nSave word element into file")
	with open(out_file_path, 'a') as out_file:
		process_bar = ShowProcess(len(content))
		count = 0
		for sentence in content:
			index, space, rest_sentence = sentence.partition(' ')

			if rest_sentence:
				result = nltk.word_tokenize(rest_sentence)
				pos = nltk.pos_tag(result)

				np_grammer = "NP: {<DT>?<JJ>*<NN|NNS>+}"  # NP NP: {<DT>? <JJ>* <NN>*}
				np_cp = nltk.RegexpParser(np_grammer)
				np_tree = np_cp.parse(pos)
				np_iob_tags = nltk.tree2conlltags(np_tree)

				vb_grammer = "V: {<V.*>}"  # Verb
				vb_cp = nltk.RegexpParser(vb_grammer)
				vb_tree = vb_cp.parse(pos)
				vb_iob_tags = nltk.tree2conlltags(vb_tree)

				p_grammer = "P: {<IN>}"  # Preposition
				p_cp = nltk.RegexpParser(p_grammer)
				p_tree = p_cp.parse(pos)
				p_iob_tags = nltk.tree2conlltags(p_tree)

				pp_grammer = "PP: {<P> <NP>}"  # PP -> P NP
				pp_cp = nltk.RegexpParser(pp_grammer)
				pp_tree = pp_cp.parse(pos)
				pp_iob_tags = nltk.tree2conlltags(pp_tree)

				vp_grammer = "VP: {<V> <NP|PP>*}"   # VP -> V (NP|PP)*
				vp_cp = nltk.RegexpParser(vp_grammer)
				vp_tree = vp_cp.parse(pos)
				vp_iob_tags = nltk.tree2conlltags(vp_tree)

				# out_file.write("Index " + index.rstrip("\n\r") + " word pairs:\n")
				if len(np_iob_tags) != len(vb_iob_tags) != len(p_iob_tags):
					print ("length mismatch")
				if len(np_iob_tags) > 1:
					for index in range(0,len(np_iob_tags)):
						label = "Other"
						if str(np_iob_tags[index][2]) != 'O':
							label = get_label(np_iob_tags[index][0], classes, count)
							out_file.write(str(np_iob_tags[index][0]) + ' ' + str(np_iob_tags[index][1]) + ' ' + str(np_iob_tags[index][2]) + ' ' + str(label) + '\n')
						elif str(vb_iob_tags[index][2]) != 'O':
							label = get_label(vb_iob_tags[index][0], classes, count)
							out_file.write(str(vb_iob_tags[index][0]) + ' ' + str(vb_iob_tags[index][1]) + ' ' + str(vb_iob_tags[index][2]) + ' ' + str(label) + '\n')
						elif str(p_iob_tags[index][2]) != 'O':
							label = get_label(p_iob_tags[index][0], classes, count)
							out_file.write(str(p_iob_tags[index][0]) + ' ' + str(p_iob_tags[index][1]) + ' ' + str(p_iob_tags[index][2]) + ' ' + str(label) + '\n')
						elif str(pp_iob_tags[index][2]) != 'O':
							label = get_label(pp_iob_tags[index][0], classes, count)
							out_file.write(str(pp_iob_tags[index][0]) + ' ' + str(pp_iob_tags[index][1]) + ' ' + str(pp_iob_tags[index][2]) + ' ' + str(label) + '\n')
						elif str(vp_iob_tags[index][2]) != 'O':
							label = get_label(vp_iob_tags[index][0], classes, count)
							out_file.write(str(vp_iob_tags[index][0]) + ' ' + str(vp_iob_tags[index][1]) + ' ' + str(vp_iob_tags[index][2]) + ' ' + str(label) + '\n')
						else:
							label = get_label(np_iob_tags[index][0], classes, count)
							out_file.write(str(np_iob_tags[index][0]) + ' ' + str(np_iob_tags[index][1]) + ' ' + str(np_iob_tags[index][2]) + ' ' + str(label) + '\n')
				out_file.write('\n')
			count = count + 1
			process_bar.show_process()
	print("\nDone")
	print("\nProcess " + str(count) + " sentences")

def dependency_parser_IOB(in_file_path, wordclass_set, out_file_path):
	print("\nRead sentence from txt")
	content = []
	with open(in_file_path) as f:
	    content = f.readlines()


	print("\nSave word element into file")
	with open(out_file_path, 'a') as out_file:
		process_bar = ShowProcess(len(content))
		count = 0
		for sentence in content:
			index, space, rest_sentence = sentence.partition(' ')

			if rest_sentence:
				result = nltk.word_tokenize(rest_sentence)
				pos = nltk.pos_tag(result)

				tree = nltk.ne_chunk(pos)
				iob_tags = nltk.tree2conlltags(tree)

				np_grammer = "NP: {<DT>?<JJ>*<NN|NNS>+}"  # NP NP: {<DT>? <JJ>* <NN>*}
				np_cp = nltk.RegexpParser(np_grammer)
				np_tree = np_cp.parse(pos)
				np_iob_tags = nltk.tree2conlltags(np_tree)

				vb_grammer = "V: {<V.*>}"  # Verb
				vb_cp = nltk.RegexpParser(vb_grammer)
				vb_tree = vb_cp.parse(pos)
				vb_iob_tags = nltk.tree2conlltags(vb_tree)

				p_grammer = "P: {<IN>}"  # Preposition
				p_cp = nltk.RegexpParser(p_grammer)
				p_tree = p_cp.parse(pos)
				p_iob_tags = nltk.tree2conlltags(p_tree)

				pp_grammer = "PP: {<P> <NP>}"  # PP -> P NP
				pp_cp = nltk.RegexpParser(pp_grammer)
				pp_tree = pp_cp.parse(pos)
				pp_iob_tags = nltk.tree2conlltags(pp_tree)

				vp_grammer = "VP: {<V> <NP|PP>*}"   # VP -> V (NP|PP)*
				vp_cp = nltk.RegexpParser(vp_grammer)
				vp_tree = vp_cp.parse(pos)
				vp_iob_tags = nltk.tree2conlltags(vp_tree)

				# out_file.write("Index " + index.rstrip("\n\r") + " word pairs:\n")
				if len(np_iob_tags) != len(vb_iob_tags) != len(p_iob_tags):
					print ("length mismatch")
				if len(np_iob_tags) > 1:
					for index in range(0,len(np_iob_tags)):
						label = "Other"
						if str(np_iob_tags[index][2]) != 'O':
							label = iob_tags[index][2]
							out_file.write(str(np_iob_tags[index][0]) + ' ' + str(np_iob_tags[index][1]) + ' ' + str(np_iob_tags[index][2]) + ' ' + str(label) + '\n')
						elif str(vb_iob_tags[index][2]) != 'O':
							label = iob_tags[index][2]
							out_file.write(str(vb_iob_tags[index][0]) + ' ' + str(vb_iob_tags[index][1]) + ' ' + str(vb_iob_tags[index][2]) + ' ' + str(label) + '\n')
						elif str(p_iob_tags[index][2]) != 'O':
							label = iob_tags[index][2]
							out_file.write(str(p_iob_tags[index][0]) + ' ' + str(p_iob_tags[index][1]) + ' ' + str(p_iob_tags[index][2]) + ' ' + str(label) + '\n')
						elif str(pp_iob_tags[index][2]) != 'O':
							label = iob_tags[index][2]
							out_file.write(str(pp_iob_tags[index][0]) + ' ' + str(pp_iob_tags[index][1]) + ' ' + str(pp_iob_tags[index][2]) + ' ' + str(label) + '\n')
						elif str(vp_iob_tags[index][2]) != 'O':
							label = iob_tags[index][2]
							out_file.write(str(vp_iob_tags[index][0]) + ' ' + str(vp_iob_tags[index][1]) + ' ' + str(vp_iob_tags[index][2]) + ' ' + str(label) + '\n')
						else:
							label = iob_tags[index][2]
							out_file.write(str(np_iob_tags[index][0]) + ' ' + str(np_iob_tags[index][1]) + ' ' + str(np_iob_tags[index][2]) + ' ' + str(label) + '\n')
				out_file.write('\n')
			count = count + 1
			process_bar.show_process()
	print("\nDone")
	print("\nProcess " + str(count) + " sentences")

if __name__ == '__main__' :

    start = time.time()

    in_file_path = "../data/sentence.txt"
    out_file_path = "../output/sentence_FrameNet.txt"
    wordclass_set = "../data/word_class.txt"

    dependency_parser_IOB(in_file_path, wordclass_set, out_file_path)

    print ("\nRunning Time: " + str(int(time.time()-start)) + 's')