#!/usr/bin/env python
# coding: utf-8

# ## 라이브러리 가져오기
# - pandas, numpy, matplotlib.pyplo, re, urllib,konlpy,tqdm 가져오기
# - tensorflow.keras.preprocessing.text.Tokenizer 가져오기
# - tensorflow, tensorflow.keras.preprocessing.sequence.pad_sequences 가져오기

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import io
import urllib.request
from konlpy.tag import Okt
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf


# ## 네이버 영화 리뷰 데이터 가져오기
# 
# - 네이버 리뷰 데이터 다운로드
# - 텍스트 파일을 DataFrame으로 만들기

# #### 네이버 리뷰 데이터 다운로드

# In[2]:


urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="review_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="review_test.txt")


# #### 텍스트 파일을 DataFrame으로 만들기

# In[3]:


train_review = pd.read_csv('review_train.txt', sep="\t")
test_review = pd.read_csv('review_test.txt', sep="\t")


# ## 데이터 EDA 및 전처리
# 
# ### 데이터 EDA
# - 학습데이터와 테스트데이터 확인하기
# - 학습데이터와 테스트데이터 개수 확인
# - Missing Value 찾고 없애기(EDA 및 전처리)
# - 리뷰 문장 단어 길이 분석(시각화 후 평균, 최소, 최고 길이)
# - 라벨(정답)의 유형 확인 (https://github.com/e9t/nsmc)
# - 학습데이터와 테스트데이터의 라벨 별 개수 확인
# 
# ### 데이터 전처리(자연어 전처리)
# - 정답 데이터 분리
# - 클리닝 (Cleaning)
# - 토큰화 (Tokenization) 
# - 불용어 제거
# - 인코딩
# - 패딩

# ### 데이터 EDA

# #### 학습데이터와 테스트데이터  확인하기
# - 학습데이터와 테스트 데이터에 어떤 값이 들어있는지 파악하기 위해 각 DataFrame를 확인하세요

# In[4]:


train_review[:10]


# In[5]:


test_review[:10]


# #### 학습데이터와 테스트데이터 개수 확인
# - 각 데이터 별 개수를 확인하세요
# - hint : len함수를 사용하세요

# In[6]:


print('학습데이터 개수 : {}'.format(len(train_review)))
print('테스트데이터 개수 : {}'.format(len(test_review))) 


# #### Missing Value(결측치) 찾고 없애기
# - 데이터의 결측치는 학습 시 문제가 생기므로 결측치를 제거하여야한다.
# 1. document열에 null이 포함된 행의 수를 찾는다.
# 2. null이 포함된 행을 확인
# 3. null이 포함된 행을 제거

# In[7]:


test_review.isnull().sum()


# In[8]:


train_review[train_review["document"].isnull()]


# In[9]:


train_review=train_review.dropna(subset=['document'])
print('train 리뷰 개수 :',len(train_review))


# In[10]:


test_review.isnull().sum()


# In[11]:


test_review[test_review["document"].isnull()]


# In[12]:


test_review["document"].isnull().value_counts()
test_review=test_review.dropna(subset=['document'])


# In[13]:


print('train 리뷰 개수 :',len(train_review))
print('test 리뷰 개수 :',len(test_review)) 


# #### 리뷰 문장 단어 길이 분석(시각화 후 평균, 최소, 최고 길이)
# - 자연어 처리에서는 단어의 길이가 학습에 중요한 요소 중 하나이다. 그러하여 자연어 처리 시 단어의 길이를 확인해야한다.
# - 시각화는 원하는 값의 대략적인 형태를 파악하기 필수이다. 
# 1. 학습데이터와 테스트데이터를 합친다.
# 2. 각 행별 리뷰의 단어 갯수를 구해 변수에 저장시킨다.
# 3. 단어의 길이의 평균을 구한다.
# 4. 시각화를 한다.
# 5. 평균, 최소, 최고 길이를 출력한다.

# In[ ]:


all_review = pd.concat([train_review,test_review],ignore_index=True)
sentence_length = all_review["document"].apply(lambda x : len(x.split()))
mean_seq_len = sentence_length.mean().astype(int)
sns.distplot(tuple(sentence_length), hist=True, kde=True, label='Sentence length')
plt.axvline(x=mean_seq_len, color='k', linestyle='--', label=f'Sequence length mean:{mean_seq_len}')
plt.title('Sentence length')
plt.legend()
plt.show()
print(f"가장 긴 문장 내 단어의 수 : {sentence_length.max()}")
print(f"가장 짧은 문장 내 단어의 수 : {sentence_length.min()}")
print(f"평균 문장 내 단어의 수 : {mean_seq_len}")


# - 라벨(정답)의 유형 확인 (https://github.com/e9t/nsmc)
# 
#     - 리뷰 데이터의 클래스은 0과 1
#     - 0: 부정적인 리뷰
#     - 1: 긍정적인 리뷰

# In[14]:


print(set(train_review["label"]))


# #### 학습데이터와 테스트데이터의 라벨 별 개수 확인
# - 클래스별 데이터의 개수가 비슷해야 지도학습 시 좋은 결과를 내놓는다. 각 클래스 별 개수가 비슷한지 확인한다.

# In[15]:


train_review['label'].value_counts()


# In[16]:


test_review['label'].value_counts()


# ### 데이터 전처리(자연어 전처리)

# ### 정답 데이터 분리
# - train과 test의 라벨 데이터를 변수에 저장

# In[17]:


train_label = np.array(train_review['label'])
test_label = np.array(test_review['label'])


# #### 클리닝 (Cleaning)
# - 방해가 되는 불필요한 문자, 기호 등을 사전에 제거하는 작업을 해야 학습 시 좋은 결과를 내놓는다.
# - 이번 시간에는 공백과 한글을 제외한 글자는 |거한다.

# In[18]:


koreanSpace_word = re.compile('[^ ㄱ-ㅣ가-힣]+')
train_review["document"]=train_review["document"].str.replace(koreanSpace_word, '')


# In[19]:


train_review


# #### 토큰화 (Tokenization)
#  - 한국어는 모델이 문맥을 더 잘 이해 할 수 있도록 의미를 내포하고 있는 가장 작은 단위인 형태소 단위로 토큰화를 진행한다.
#  - 그 이유는 한국어는 영어처럼 띄어쓰기 단위로 토큰화를 진행 시 의미를 갖는 단어로 토큰화가 어렵기 때문이다.
#  - 그 예시는 "우리는", "우리에게", "우리를"과 같은 단어를 다른 단어로 인식해 자연어 처리하기가 어렵다.
#  
# - Okt를 사용하여 형태소분석을 한다.
#  

# In[20]:


okt = Okt()
print(okt.morphs("아버지가방에 들어가신다"))


# In[21]:


train_tok_sentence = []
for sentence in tqdm(train_review['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True)
    train_tok_sentence.append(tokenized_sentence)


# #### 불용어 제거
# - 큰 의미가 없는 단어를 제거함으로써 문맥적으로 모델이 더 잘 이해 할 수 있도록 만들어준다.
# 1. 불용어사전을 list형태로 만든다
# 2. 각 단어가 불용어 사전에 있으면 제거한다.

# In[22]:


stopWord = []
with io.open("불용어사전.txt", mode='r',encoding="utf-8") as f:
    for a in f:
        stopWord.append(a.strip())
print(stopWord)


# In[23]:


train_remove_tok_sentence=[]
for sentence in train_tok_sentence:
    train_remove_tok_sentence.append([x for x in sentence if not x in stopWord])


# In[24]:


train_remove_tok_sentence[:5]


# In[25]:


zero_length_index = []
for idx, sentence in enumerate(train_remove_tok_sentence):
    if len(sentence) == 0:
        zero_length_index.append(idx)


# In[26]:


train_remove_tok_sentence = np.delete(train_remove_tok_sentence, zero_length_index, axis=0)
train_label = np.delete(train_label, zero_length_index, axis=0)


# #### 인코딩
# - 단어를 정수로 바꿔야지 모델을 더 잘 학습시킬 수 있다.
# 1. Tokenizer()를 변수로 넣기 
# 2. 전처리된 훈련데이터를 tokenizer에 fit하기
# 3. 훈련데이터를 인코딩하기

# In[27]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_remove_tok_sentence)
train_encoding_sentence = tokenizer.texts_to_sequences(train_remove_tok_sentence)


# In[28]:


vocab_size=len(tokenizer.word_index)+1


# In[29]:


train_encoding_sentence[:5]


# #### 패딩
# - 모든 shape를 똑같게 하기 위해서 남는 자리에 0을 넣는다.
# 1. 최고 길이 구하기
# 2. padding 하기

# In[30]:


max_length = 0
for sentence in train_encoding_sentence:
    if max_length < len(sentence):
        max_length = len(sentence)
print(max_length)


# In[31]:


train_padding_sentence = pad_sequences(train_encoding_sentence, maxlen = max_length)


# In[32]:


train_padding_sentence[:5]


# In[33]:


train_padding_sentence = pad_sequences(train_encoding_sentence, maxlen = max_length)


# ## 긍정리뷰인지 부정리뷰인지 분류하는 모델 만들기
# 
# - embedding_dim 설정(embedding_dim : 임베딩의 치수)
# - rnn 딥러닝 알고리즘을 활용하여 분류 모델 만들기

# In[34]:


embedding_dim = 128

model_rnn = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.SimpleRNN(128),
    tf.keras.layers.Dense(1, activation="sigmoid")
])


# In[35]:


model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])


# In[36]:


filename = 'fit-model-tmp-chkpoint.h5'
EarlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
checkpoint = tf.keras.callbacks.ModelCheckpoint(filename, verbose=1, monitor='val_loss', save_best_only=True, mode='auto')
history=model_rnn.fit(train_padding_sentence,train_label,epochs = 10, callbacks=[checkpoint,EarlyStop],batch_size=128,validation_split=0.2)


# ### 테스트데이터 전처리
# - 위에서 훈련데이터를 전처리 하는  방식이랑 비슷하게 테스트데이터를 전처리
# 1. 테스트데이터 전처리
# 2. 테스트데이터를 활용해서 모델 정확도 확인하기

# In[37]:


test_review["document"]=test_review["document"].str.replace(koreanSpace_word, '')
test_tok_sentence = []
for sentence in tqdm(test_review['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True)
    test_tok_sentence.append(tokenized_sentence)
test_remove_tok_sentence=[]
for sentence in test_tok_sentence:
    test_remove_tok_sentence.append([x for x in sentence if not x in stopWord])
zero_length_index = []
for idx, sentence in enumerate(test_remove_tok_sentence):
    if len(sentence) == 0:
        zero_length_index.append(idx)
test_remove_tok_sentence = np.delete(test_remove_tok_sentence, zero_length_index, axis=0)
test_label = np.delete(test_label, zero_length_index, axis=0)
test_encoding_sentence = tokenizer.texts_to_sequences(test_remove_tok_sentence)
test_padding_sentence = pad_sequences(test_encoding_sentence, maxlen = max_length)


# In[38]:


print("\n 테스트 정확도:" + str(int(model_rnn.evaluate(test_padding_sentence, test_label)[1]*100))+"%")


# In[ ]:




