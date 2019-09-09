
import urllib.request
import os
import tarfile
import re
import numpy as np
import pandas as pd
import string

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

def calc_logistic_score(pv,train_y,transform_pv,test_y):
    log_reg = LogisticRegression(random_state=2019,solver='lbfgs')
    # Train.
    log_reg.fit(pv, train_y)
    pred_train = log_reg.predict(pv)
    pred_prob_train = log_reg.decision_function(pv)
    # Test.
    pred_test = log_reg.predict(transform_pv)
    pred_prob_test = log_reg.decision_function(transform_pv)
    # Compute classification performance.
    train_accuracy = accuracy_score(train_y, pred_train)
    test_accuracy = accuracy_score(test_y, pred_test)
    train_auc = roc_auc_score(train_y, pred_prob_train)
    test_auc = roc_auc_score(test_y, pred_prob_test)
    return test_accuracy,test_auc

# Util functions
# Normalize text
def normalize_text(texts, stops=None):
    # Lower case
    texts = [x.lower() for x in texts]

    # EOL
    texts = [re.compile(r"\\n").sub(' ',x) for x in texts]
    
    # Replace punctuation with space
    texts = [x.translate(x.maketrans(string.punctuation, ' '*len(string.punctuation))) for x in texts]
    
    # Remove numbers
    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
    
    if stops:
        # Remove stopwords
        texts = [' '.join([word for word in x.split() if word not in (stops)]) for x in texts]

    # split into list
    texts = [x.split() for x in texts]
    
    return(texts)

def rm_tags(text):
	re_tag = re.compile(r'<[^>]+>')
	return re_tag.sub('',text)

def read_files(filetype):
	path = "data/aclImdb/"
	file_list=[]
	
	positive_path=path + filetype+"/pos/"
	for f in os.listdir(positive_path):
		file_list+=[positive_path+f]
		
	negative_path=path + filetype+"/neg/"
	for f in os.listdir(negative_path):
		file_list+=[negative_path+f]
		
	print('read',filetype,'files:',len(file_list))
	
	all_labels = ([1]*12500+[0]*12500)
	all_texts = []
	for fi in file_list:
		with open(fi,encoding='utf8') as file_input:
			all_texts += [rm_tags("".join(file_input.readlines()))]
	
	return all_labels,all_texts

def download_Imdb_data():
	""" download """
	url="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
	filepath="data/Large Movie Review Dataset/aclImdb_v1.tar.gz"
	if not os.path.isfile(filepath):
		result=urllib.request.urlretrieve(url,filepath)
		print('downloaded:',result)

	""" extract """
	if not os.path.exists("data/Large Movie Review Dataset/aclImdb"):
		tfile = tarfile.open("data/Large Movie Review Dataset/aclImdb_v1.tar.gz",'r:gz')
		result = tfile.extractall('data/')

	""" train & test """
	y_train,train_text = read_files("train")
	y_test,test_text = read_files("test")
	np_train_text = np.array(train_text)
	np_train_y = np.array(y_train)
	np_test_text = np.array(test_text)
	np_test_y = np.array(y_test)

	train = pd.DataFrame(np.array([np_train_text,np_train_y]).T,columns=['text','label'])
	test = pd.DataFrame(np.array([np_test_text,np_test_y]).T,columns=['text','label'])

	""" write to file """
	train.to_csv('data/Large Movie Review Dataset/train.csv',index=False)
	test.to_csv('data/Large Movie Review Dataset/test.csv',index=False)


def load_Imdb_data():
    train = pd.read_csv('data/Large Movie Review Dataset/train.csv')
    test = pd.read_csv('data/Large Movie Review Dataset/test.csv')
    train_X = train['text'].values.tolist()
    train_y = train['label'].values.tolist()
    test_X = test['text'].values.tolist()
    test_y = test['label'].values.tolist()
    return train,test

def load_SST_data():
    train = pd.read_csv('data/Stanford Sentiment Treebank/train.csv')
    test = pd.read_csv('data/Stanford Sentiment Treebank/test.csv')
    train_X = train['text'].values.tolist()
    train_y = train['label'].values.tolist()
    test_X = test['text'].values.tolist()
    test_y = test['label'].values.tolist()
    return train,test

def load_Yelp_data():
    train = pd.read_csv('data/Yelp Review Binary Classification/train.csv',header=None,names=['text','label'])
    test = pd.read_csv('data/Yelp Review Binary Classification/test.csv',header=None,names=['text','label'])
    train_X = train['text'].values.tolist()
    train_y = train['label'].values.tolist()
    test_X = test['text'].values.tolist()
    test_y = test['label'].values.tolist()
    return train,test
