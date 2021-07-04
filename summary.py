import os
import re
import string
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from spacy.lang.ro.stop_words import STOP_WORDS
from nltk.tokenize import sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import spacy

lemm = spacy.load('ro_core_news_sm')
 

start_time = time.time()

# corolaFile = "word2vec_Oftalmologie_400_1_10_final.txt"
# dim=400

stop = list(STOP_WORDS)
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
 
def rem_ascii(s):
    return "".join(s)
 
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    processed = re.sub(r"\d+","",punc_free)
    return processed
 
def loadCorolaModel(corolaFile):
    word_embeddings = {}
    f = open(corolaFile, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    return word_embeddings

def lemmaSentences(cleanSentences):
    lemmaSentences = []
    for sentence in cleanSentences:
        sent = ''
        lemm1 = lemm(sentence.lower())
        for l in lemm1:
            sent += " " + l.lemma_
        lemmaSentences.append(sent)
    return lemmaSentences
 
# word_embeddings = loadCorolaModel(corolaFile)

# print("Vocab Size = ",len(word_embeddings))

def algorithm(text, name, dimens , proc):
    word_embeddings = loadCorolaModel(name)
    dim = dimens

    sentences = sent_tokenize(text)
    cleaned_texts = [rem_ascii(clean(sentence)) for sentence in sentences]
    lemma_Sentences = lemmaSentences(cleaned_texts)

    sentence_vectors = []
    for i in lemma_Sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((dim,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((dim,))
        sentence_vectors.append(v)


    sim_mat = np.zeros([len(lemma_Sentences), len(lemma_Sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,dim),sentence_vectors[j].reshape(1,dim))[0,0]
    sim_mat = np.round(sim_mat,3)
    print(sim_mat)
    

    nx_graph = nx.from_numpy_array(sim_mat)
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(nx_graph)
    nx.draw(nx_graph, with_labels=True, font_weight='bold')
    nx.draw_networkx_edge_labels(nx_graph,pos,font_color='red')


    scores = nx.pagerank(nx_graph)

    proc = proc / 100


    ranked_sentences = sorted(((scores[i],i) for i,s in enumerate(sentences)), reverse=True)
    arranged_sentences = sorted(ranked_sentences[0:int(len(sentences) * proc)], key=lambda x:x[1])
    print("\n".join([sentences[x[1]] for x in arranged_sentences]))
    print("--- %s seconds ---" % (time.time() - start_time))

    return ("\n".join([sentences[x[1]] for x in arranged_sentences]))

