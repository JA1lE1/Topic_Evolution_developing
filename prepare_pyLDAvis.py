# -*- coding: utf-8 -*-

'''
pyLDAvis 缺少doc_length 和term_frequency的组件
data['doc_lengths'] = doc_lengths 就是每篇文章的长度
data['term_frequency'] = term_frequency 至于词频应该在counts那个地方就有保存成np的格式就可以

'''

#更换data_prepare中的csv文件进行对应不同预料的训练

import jieba
from collections import Counter
from tqdm import tqdm
import re
#import codecs
# from hanziconv import HanziConv
from gensim.corpora import WikiCorpus
import time
import numpy as np
#import warnings
#warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora, models
# 提前下好， gensim的wiki 词向量训练
import multiprocessing

from data_prepare import news_prepare

def _count_unique_tokens(tokenized_docs):
    tokens = []
    for i, doc in tokenized_docs:
        tokens += doc
    return Counter(tokens)

# 这个函数的功能是将 具体每篇文档 编码成字典中对应的序号
def _encode(tokenized_docs, encoder):
    return [(i, [encoder[t] for t in doc]) for i, doc in tokenized_docs]


def _remove_tokens(tokenized_docs, counts, min_counts, max_counts):
    """
    Words with count < min_counts or count > max_counts
    will be removed.
    """
    total_tokens_count = sum(
        count for token, count in counts.most_common()
    )
    print('total number of tokens:', total_tokens_count)

    unknown_tokens_count = sum(
        count for token, count in counts.most_common()
        if count < min_counts or count > max_counts
    )
    print('number of tokens to be removed:', unknown_tokens_count)

    keep = {}
    for token, count in counts.most_common():
        keep[token] = count >= min_counts and count <= max_counts

    return [(i, [t for t in doc if keep[t]]) for i, doc in tokenized_docs]


def _create_token_encoder(counts):

    total_tokens_count = sum(
        count for token, count in counts.most_common()
    )
    print('total number of tokens:', total_tokens_count)

    encoder = {}
    decoder = {}
    word_counts = []
    i = 0

    for token, count in counts.most_common():
        # counts.most_common() is in decreasing count order
        encoder[token] = i
        decoder[i] = token
        word_counts.append(count)
        i += 1

    return encoder, decoder, word_counts

def get_windows(doc, hws=5):
    """
    For each word in a document get a window around it.

    Arguments:
        doc: a list of words.
        hws: an integer, half window size.

    Returns:
        a list of tuples, each tuple looks like this
            (word w, window around w),
            window around w equals to
            [hws words that come before w] + [hws words that come after w],
            size of the window around w is 2*hws.
            Number of the tuples = len(doc).
    """
    length = len(doc)
    print(length)
    assert length > 2*hws, 'doc is too short!'

    inside = [(w, doc[(i - hws):i] + doc[(i + 1):(i + hws + 1)])
              for i, w in enumerate(doc[hws:-hws], hws)]

    # for words that are near the beginning or
    # the end of a doc tuples are slightly different
    beginning = [(w, doc[:i] + doc[(i + 1):(2*hws + 1)])
                 for i, w in enumerate(doc[:hws], 0)]

    end = [(w, doc[-(2*hws + 1):i] + doc[(i + 1):])
           for i, w in enumerate(doc[-hws:], length - hws)]

    return beginning + inside + end

#----------------------------------------------------------------------
def data_prepare():
    """"""
    # 提前下好， gensim的wiki 词向量训练
    
    # article_num = 0   ## 预先设置训练检测


    # wiki = WikiCorpus('zhwiki-latest-pages-articles.xml.bz2',lemmatize=False, dictionary={})
    
    stopwords = [line.strip() for line in open('stopwords.txt', 'r', encoding='utf-8').readlines()]
    #with open('stopwords.txt','r',encoding='utf8').readlines() as f:
        #stopwords = [ w.strip() for w in f] 
    #stopwords = codecs.open('stopwords.txt','r',encoding='utf8').readlines()
    #stopwords = [ w.strip() for w in stopwords ] 
    # start = time.time()
    for text in news_prepare('cn_sina.csv'):
        # text = ' '.join(text)
        # # text = HanziConv.toSimplified(text)
        # #re.sub('[：·•’!\"#$%&\'()*+，,-./:：;；<=>?@，。?★、…【】《》？“”〞‘’！[\\]^_`{}（）~]+', "", text)
        # text = text.strip()
        # print(text)
        seg_list = list(jieba.cut(text))
        # ['歐幾里', '得', ' ', '西元前', '三世', '紀的', '古希臘', '數學家', ' ', '現在', '被', '認
        #  '是', '幾何', '之父', ' ', '此畫', '為拉斐爾', '的', '作品', ' ', '雅典', '學院']
        new_text = [x for x in seg_list  if  (re.compile(u'[\u4e00-\u9fa5]+').search(x) or \
                        re.compile("[\"”“，？?\,\.。,0-9]+").search(x)) and (x not in stopwords)]
        #new_text = [x for x in seg_list if     re.compile('[^a-zA-Z]+').search(x) and x != ' '] ## 原来的版本是len(x)  > 1 这里 不能这样8行
        
        # article_num = article_num + 1
        # if article_num == 10:
        #     break        
        yield new_text
    

if __name__ == '__main__':
    min_counts = 20
    #MAX_COUNTS = 1800
    max_counts = 1800
    # words with count < MIN_COUNTS
    # and count > MAX_COUNTS
    # will be removed
    
    #MIN_LENGTH = 15
    min_length = 15
    
    # start = time.time()

    ##
    #docs = [(i, doc) for i, doc in enumerate(docs)]
    tokenized_docs = [(i, doc) for i, doc in enumerate(data_prepare())]
    
    
    # remove short documents
    n_short_docs = sum(1 for i, doc in tokenized_docs if len(doc) < min_length)
    tokenized_docs = [(i, doc) for i, doc in tokenized_docs if len(doc) >= min_length]
    print('number of removed short documents:', n_short_docs)

    # remove some tokens
    counts = _count_unique_tokens(tokenized_docs)
    tokenized_docs = _remove_tokens(tokenized_docs, counts, min_counts, max_counts)
    n_short_docs = sum(1 for i, doc in tokenized_docs if len(doc) < min_length)
    tokenized_docs = [(i, doc) for i, doc in tokenized_docs if len(doc) >= min_length]
    print('number of additionally removed short documents:', n_short_docs)
    
    
    #tokenized_docs = [(i, doc) for i, doc in tqdm(docs)]             ## 注意这里和enumerate的区别？
    counts = _count_unique_tokens(tokenized_docs)                    ## 返回所有的token在一个Counter类型中
    ## print出token的个数（包括重复的个数）返回值:encoder 和decoder就是字面的意思 
    ## 词->序号， 序号-> 词， 类型都是字典,word_counts与上一个函数一样是所有的token的数量
    encoder, decoder, word_counts = _create_token_encoder(counts)    ## print出token的个数（包括重复的个数）返回值:encoder 和decoder就是字面的意思 词->序号， 序号-> 词， 类型都是字典,word_counts与上一个函数一样是所有的token的数量

    encoded_docs = _encode(tokenized_docs, encoder)
    doc_lengths = np.zeros(len(encoded_docs), dtype='int32')
    for (i,_doc) in enumerate(encoded_docs):
        doc_lengths[i] = len(np.unique(_doc[1])) 

    np.save('doc_lengths.npy', doc_lengths)

    term_frequency = np.array(word_counts, dtype='int32')
    np.save('term_frequency.npy', term_frequency)




#     print('\nminimum word count number:', word_counts[-1])
#     print('this number can be less than MIN_COUNTS because of document removal')
    
#     encoded_docs = _encode(tokenized_docs, encoder)
    
#     doc_decoder = {i: doc_id for i, (doc_id, doc) in enumerate(encoded_docs)}    # 由于我没有删除任何文本 所以我觉得这个毫无意义
    
    
#     data = []
#     # new ids are created here
#     HALF_WINDOW_SIZE =5
#     for index, (_, doc) in tqdm(enumerate(encoded_docs)):
#         windows = get_windows(doc, HALF_WINDOW_SIZE)
#         # index represents id of a document, 
#         # windows is a list of (word, window around this word),
#         # where word is in the document
#         data += [[index, w[0]] + w[1] for w in windows]
    
#     data = np.array(data, dtype='int64')
    
#     word_counts = np.array(word_counts)
#     unigram_distribution = word_counts/sum(word_counts)
    
    
    
#     vocab_size = len(decoder)
#     embedding_dim = 50
    
#     # train a skip-gram word2vec model
#     texts = [[str(j) for j in doc] for i, doc in encoded_docs]
#     #texts = [[decoder[j] for j in doc] for i, doc in encoded_docs]
# #     model = models.Word2Vec(texts, size=embedding_dim, min_count=1)
#     model = models.Word2Vec(texts, size=embedding_dim, window=5, sg=1, negative=15, min_count=1, workers=multiprocessing.cpu_count())
#     model.init_sims(replace=True)
    
#     word_vectors = np.zeros((vocab_size, embedding_dim)).astype('float32')
#     for i in decoder:
#         word_vectors[i] = model.wv[str(i)]
    
    
#     texts = [[decoder[j] for j in doc] for i, doc in encoded_docs]
#     dictionary = corpora.Dictionary(texts)
#     corpus = [dictionary.doc2bow(text) for text in texts]
    
    
#     n_topics = 25  ## 超参数
#     lda = models.LdaModel(corpus, alpha=0.9, id2word=dictionary, num_topics=n_topics)
#     corpus_lda = lda[corpus]
    
#     doc_weights_init = np.zeros((len(corpus_lda), n_topics))
#     for i in tqdm(range(len(corpus_lda))):
#         topics = corpus_lda[i]
#         for j, prob in topics:
#             doc_weights_init[i, j] = prob
            
            
#     np.save('data.npy', data)
#     np.save('word_vectors.npy', word_vectors)
#     np.save('unigram_distribution.npy', unigram_distribution)
#     np.save('decoder.npy', decoder)
#     np.save('doc_decoder.npy', doc_decoder)
#     np.save('doc_weights_init.npy', doc_weights_init)
    
#     # end = time.time()
#     # print(start - end)
#     # print('success!')
    
