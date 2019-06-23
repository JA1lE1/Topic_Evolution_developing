import torch
import numpy as np
import topics

def softmax(x):
    # x has shape [batch_size, n_classes]
    e = np.exp(x)
    n = np.sum(e, 1, keepdims=True)
    return e/n


state = torch.load('model_state.pytorch', map_location=lambda storage, loc: storage)
n_topics = 25

doc_weights = state['doc_weights.weight'].cpu().clone().numpy()
topic_vectors = state['topics.topic_vectors'].cpu().clone().numpy()
resulted_word_vectors = state['neg.embedding.weight'].cpu().clone().numpy()

# distribution over the topics for each document
topic_dist = softmax(doc_weights)

# vector representation of the documents
doc_vecs = np.matmul(topic_dist, topic_vectors)

# 单词字典对应数字
decoder = np.load('decoder.npy')[()]

data = topics.prepare_topics(doc_weights, topic_vectors, resulted_word_vectors, decoder)



doc_lengths = np.load("doc_lengths.npy")
term_frequency = np.load("term_frequency.npy")
data['doc_lengths'] = doc_lengths
data['term_frequency'] = term_frequency
np.savez('topics.pyldavis', **data)