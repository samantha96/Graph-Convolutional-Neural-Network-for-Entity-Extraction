# Graph-Convolutional-neural-network
Graph Convolutional Neural Network for Entity Extraction

Information Extraction (IE) is the process of extracting structured information from unstructured machine-readable documents 

In Diego Marcheggiani and Ivan Titov’s paper Encoding Sentences with Graph Convolutional Networks for Semantic Role Labeling, researchers found that the results of syntactic analysis and the results of semantic role labeling are mostly mirrored. Based on the above observations, researchers proposed a graph-based convolutional network(GCN) method for semantic role labeling. Use GCN to encode a syntax-dependent tree, resulting in a potential feature representation of the word in the sentence.

Our research group used CoNLL 2003 as our dataet and tried to use GCN to do Named Entity Extraction here. With the help of Stanford Parser and word2vec, we got the three matrix to feed in GCN model. Our research group used python to extract them and build them into matrix. After the preprocessing, we devided the dataset into 2 part, the first 50% as training set and the last 50% as testing set. In order to test the behavior of the GCN model, this project implemented a complete GCN model using python and tensorflow. Finally the program got 86.02% of accuracy on the testing set.

Although we got 86.02% of accuracy for our model,which is even higher than the state-of-art approach, we still want to think further about how to improve our model. One problem for our model is that all sentences used to train our model are from CoNLL 2003, which collected sentences from newspapers and magazines. We can try more different kinds of tagged datasets to train our model in order to increase its ability to adapt new datasets. In addition, We can also think that how the length and complexity of each sentence affect the accuracy of the model. Collecting longer and more complex sentences with correct tags might help us to train a more accurate model.

# Data Processing
To feed GCN model and generate the neural network, we must provide three matrices.
1. the dependencies between words in a sentence
2. features of each word
3. class of each word
