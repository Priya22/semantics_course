#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import pandas as pd
import numpy as np
import os, re, sys


# In[2]:


from nltk.corpus import brown


# In[3]:


from nltk.util import ngrams


# In[4]:


from collections import Counter, defaultdict


# In[5]:


import sklearn


# In[6]:


from scipy import sparse


# In[7]:


nltk.download('brown')


# In[8]:


words = brown.words()


# In[9]:


corpus = ' '.join(words)


# In[10]:


corpus[:100]


# In[11]:


type(words)


# In[12]:


words = list(words)


# In[13]:


words[:10]


# In[14]:


len(words)


# In[15]:


unigram_counter = Counter(list(words))


# In[16]:


unigram_counter.most_common(10)


# In[17]:


top_500 = unigram_counter.most_common(5000)


# In[18]:


W = [x[0] for x in top_500]


# In[19]:


print("Most common: ", W[:5])
print("Least common: ", W[-5:])


# In[20]:


#Add RG65 words
rg_data = [('cord', 'smile', 0.02), ('rooster', 'voyage', 0.04), ('noon', 'string', 0.04), ('fruit', 'furnace', 0.05),
           ('autograph', 'shore', 0.06), ('automobile', 'wizard', 0.11), ('mound', 'stove', 0.14), ('grin', 'implement', 0.18),
           ('asylum', 'fruit', 0.19), ('asylum', 'monk', 0.39), ('graveyard', 'madhouse', 0.42), ('glass', 'magician',0.44),
          ('boy', 'rooster', 0.44), ('cushion', 'jewel', 0.45), ('monk', 'slave', 0.57), ('asylum', 'cemetery', 0.79), 
          ('coast', 'forest', 0.85), ('grin', 'lad', 0.88), ('shore', 'woodland', 0.90), ('monk', 'oracle', 0.91), ('boy', 'sage', 0.96),
          ('automobile', 'cushion', 0.97), ('mound','shore',0.97), ('lad', 'wizard', 0.99), ('forest', 'graveyard', 1.0), 
          ('food', 'rooster', 1.09), ('cemetery', 'woodland', 1.18), ('shore', 'voyage', 1.22), ('bird', 'woodland', 1.24),
          ('coast', 'hill', 1.26), ('furnace', 'implement', 1.37), ('crane', 'rooster', 1.41), ('hill', 'woodland', 1.48),
          ('car', 'journey', 1.55), ('cemetery', 'mound', 1.69), ('glass', 'jewel', 1.78), ('magician', 'oracle', 1.82), 
          ('crane', 'implement', 2.37), ('brother', 'lad', 2.41), ('sage', 'wizard', 2.46), ('oracle', 'sage', 2.61), 
          ('bird', 'crane', 2.63), ('bird', 'cock', 2.63), ('food', 'fruit', 2.69), ('brother', 'monk', 2.74), ('asylum', 'madhouse', 3.04),
          ('furnace', 'stove', 3.11), ('magician', 'wizard', 3.21), ('hill', 'mound', 3.29), ('cord', 'string', 3.41), 
          ('glass', 'tumbler', 3.45), ('grin', 'smile', 3.46), ('serf', 'slave', 3.46), ('journey', 'voyage', 3.58), 
          ('autograph', 'signature', 3.59), ('coast', 'shore', 3.60), ('forest', 'woodland', 3.65), ('implement', 'tool', 3.66),
          ('cock', 'rooster', 3.68), ('boy', 'lad', 3.82), ('cushion', 'pillow', 3.84), ('cemetery', 'graveyard', 3.88), 
          ('automobile', 'car', 3.92), ('midday', 'noon', 3.94), ('gem', 'jewel', 3.94)] 


# In[21]:


rg_45_words = []
for s in rg_data:
    rg_45_words.extend([s[0], s[1]])


# In[22]:


rg_45_words = list(set(rg_45_words))


# In[23]:


print(len(rg_45_words), rg_45_words[:5])


# In[24]:


W.extend(rg_45_words)
W = list(set(W))


# In[25]:


len(W)


# In[26]:


#dictionary
W_ind = {}
i = 0
for w in W:
    W_ind[w] = i
    i = i+1


# In[27]:


#bigram counts
n = 2
bigrams = ngrams(brown.words(), n)


# In[28]:


brown.words()


# In[29]:


type(bigrams)


# In[30]:


bigrams = list(bigrams)


# In[31]:


bigrams[:5]


# In[32]:


cc_mat = np.zeros((len(W), len(W)))


# In[33]:


cc_mat.shape


# In[34]:


for big in bigrams:
    try:
        cc_mat[W_ind[big[0]]][W_ind[big[1]]] += 1
    except KeyError as e:
        pass


# In[35]:


cc_mat[0]


# In[36]:


cc_mat = sparse.csr_matrix(cc_mat)


# In[37]:


cc_mat


# In[38]:


#ppmi
def get_ppmi(cc_mat):
    row_sum = np.sum(cc_mat, axis=1)
    col_sum = np.sum(cc_mat, axis = 0)
    total_num = np.sum(row_sum)
    
    denom = np.outer(row_sum, col_sum) / total_num
    
    probs = cc_mat / denom
    
    probs = np.nan_to_num(probs)
    
            
    #probs = np.divide(cc_mat, denom, out=np.zeros_like(cc_mat.shape), where=denom!=float(0))
    with np.errstate(divide='ignore'):
        probs = np.log2(probs)
    
    probs[np.isinf(probs)] = 0.0
    
    probs[probs < 0] = 0.0
    
    return probs
    
    
    


# In[39]:


cc_mat


# In[40]:


#np.zeros_like(cc_mat)


# In[41]:


ppmi_mat = get_ppmi(cc_mat)


# In[42]:


ppmi_mat.shape


# In[44]:


np.min(ppmi_mat)


# In[46]:


np.max(ppmi_mat)


# In[48]:


from sklearn.decomposition import PCA


# In[49]:


pca_10 = PCA(n_components = 10)
pca_100 = PCA(n_components = 100)
pca_300 = PCA(n_components = 300)


# In[50]:


pca_10_mat = pca_10.fit_transform(ppmi_mat)


# In[52]:


pca_100_mat = pca_100.fit_transform(ppmi_mat)
pca_300_mat = pca_300.fit_transform(ppmi_mat)


# In[53]:


pca_10_mat.shape, pca_100_mat.shape, pca_300_mat.shape


# In[54]:


S = []
S_wordpairs = []
for pair in rg_data:
    if (pair[0] in W) and (pair[1] in W):
        S_wordpairs.append((pair[0], pair[1]))
        S.append(pair[2])


# In[55]:


len(S)


# In[56]:


from sklearn.metrics.pairwise import cosine_similarity


# In[59]:


np.squeeze(np.asarray(ppmi_mat[0]))


# In[60]:


pca_100_mat[0]


# In[66]:


cosine_similarity(ppmi_mat[1],ppmi_mat[2])[0][0]


# In[65]:


cosine_similarity([pca_10_mat[1]],[pca_10_mat[2]])


# In[67]:


s_m1 = []
s_m10 = []
s_m100 = []
s_m300 = []

for i, pair in enumerate(S_wordpairs):
    word_1, word_2 = pair
    ind_1, ind_2 = W_ind[word_1], W_ind[word_2]
    
    #ppmi
    s_m1.append(cosine_similarity(ppmi_mat[ind_1],ppmi_mat[ind_2])[0][0])
    
    s_m10.append(cosine_similarity([pca_10_mat[ind_1]], [pca_10_mat[ind_2]])[0][0])
    
    s_m100.append(cosine_similarity([pca_100_mat[ind_1]], [pca_100_mat[ind_2]])[0][0])
    
    s_m300.append(cosine_similarity([pca_300_mat[ind_1]], [pca_300_mat[ind_2]])[0][0])


# In[68]:


len(s_m1), len(s_m10), len(s_m100), len(s_m300)


# In[72]:


np.sum(s_m300)


# In[74]:


from scipy.stats import pearsonr


# In[75]:


pr_m1 = pearsonr(S, s_m1)
print(pr_m1)


# In[76]:


pr_m10 = pearsonr(S, s_m10)
print(pr_m10)


# In[77]:


pr_m100 = pearsonr(S, s_m100)
print(pr_m100)


# In[78]:


pr_m300 = pearsonr(S, s_m300)
print(pr_m300)


# In[ ]:




