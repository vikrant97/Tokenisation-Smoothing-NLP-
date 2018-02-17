import re
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from random import randint
###TOKENIZE THE INPUT####
def tokenize(sentence):
	tokens=re.findall('[A-Z]?[a-z]+|[A-Z]+|[0-9]+th|[0-9]+st|[0-9]+rd|[0-9]+nd|[a-z]+-[a-z]+|Dr\.|Mr\.|Mrs\.|\'s|\'d|\.|,|&|\d{1,3}\.\d{1,3}\.\d{1,3}|[\w\.-]+@[\w\.\w]|http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|&',sentence)
	return tokens

####UNIGRAM MODEL#######
def get_unigrams(corpus):
	unigrams={}
	for sentence in corpus.split('\n'):
		tokens=tokenize(sentence)
		for k in tokens:
			if k in unigrams:
				unigrams[k]+=1
			else:
				unigrams[k]=1
	return unigrams

######BIGRAM MODEL#########
def get_bigrams(corpus):
	bigrams={}
	for sentence in corpus.split('\n'):
		tokens=tokenize(sentence)
		i=0
		length=len(tokens)-1
		while i<length:
			if tokens[i] not in bigrams:
				bigrams[tokens[i]]={}
			if tokens[i+1] not in bigrams[tokens[i]]:
				bigrams[tokens[i]][tokens[i+1]]=1
			else:
				bigrams[tokens[i]][tokens[i+1]]+=1
			i+=1
	return bigrams

#####TRIGRAM MODEL#####
def get_trigrams(corpus):
	trigrams={}
	for sentence in corpus.split('\n'):
		tokens=tokenize(sentence)
		i=0
		length=len(tokens)-2
		while i<length:
			if tokens[i] not in trigrams:
				trigrams[tokens[i]]={}
			if tokens[i+1] not in trigrams[tokens[i]]:
				trigrams[tokens[i]][tokens[i+1]]={}
			if tokens[i+2] not in trigrams[tokens[i]][tokens[i+1]]:
				trigrams[tokens[i]][tokens[i+1]][tokens[i+2]]=1
			else:
				trigrams[tokens[i]][tokens[i+1]][tokens[i+2]]+=1
			i+=1
	return trigrams

def p_unigram(unigrams):
	uni_prob=[]
	total=0
	for w in unigrams:
		total+=unigrams[w]
	for w in unigrams:
		uni_prob.append([w,unigrams[w]/float(total)])
	return uni_prob

def p_bigram(unigrams,bigrams):
	bigram_prob=[]
	for w1 in bigrams:
		for w2 in bigrams[w1]:
			bigram_prob.append([w1,w2,bigrams[w1][w2]/float(unigrams[w1])])
	return bigram_prob

def p_trigram(bigrams,trigrams):
	trigram_prob=[]
	for w1 in trigrams:
		for w2 in trigrams[w1]:
			for w3 in trigrams[w1][w2]:
				trigram_prob.append([w1,w2,w3,trigrams[w1][w2][w3]/float(bigrams[w1][w2])])
	return trigram_prob

def laplace_unigrams(unigrams,V):
    uni_prob = []
    N = sum(unigrams.values())
    for word in unigrams:
        uni_prob.append([word,(unigrams[word] + 1)/ float(N+V)])
    return uni_prob

def laplace_bigrams(unigrams,bigrams,V):
	bigram_prob=[]
	for w1 in bigrams:
		for w2 in bigrams[w1]:
			bigram_prob.append([w1,w2,(bigrams[w1][w2]+1)/float(unigrams[w1]+V)])
	return bigram_prob

def laplace_trigrams(bigrams,trigrams,V):
	trigram_prob=[]
	for w1 in trigrams:
		for w2 in trigrams[w1]:
			for w3 in trigrams[w1][w2]:
				trigram_prob.append([w1,w2,w3,(trigrams[w1][w2][w3]+1)/float(bigrams[w1][w2]+V)])
	return trigram_prob

def witten_bell1(unigrams,bigrams):
	wb_prob=[]
	for w1 in bigrams:
		for w2 in bigrams[w1]:
			ngram_count = bigrams[w1][w2]
			prior_count = unigrams[w1] 
			type_count = len(bigrams[w1])
			vocab_size = sum([len(bigrams[key]) for key in bigrams])
			z = vocab_size - type_count
			if ngram_count == 0:
				prob = float(type_count)/float(z*(prior_count + type_count))
			else:
				prob = float(ngram_count)/float(prior_count + type_count)
			wb_lambda=1-bigrams[w1][w2]/float(bigrams[w1][w2]+sum((bigrams[w1]).values()))
			prob=(wb_lambda)*prob+(1-wb_lambda)*unigrams[w2]/float(sum(unigrams.values()))
			wb_prob.append([w1,w2,prob])
	return wb_prob

def witten_bell2(unigrams,bigrams,trigrams):
	wb_prob=[]
	for w1 in trigrams:
		for w2 in trigrams[w1]:
			for w3 in trigrams[w1][w2]:
				ngram_count = trigrams[w1][w2][w3]
				prior_count = bigrams[w1][w2] 
				type_count = len(trigrams[w1][w2])
				vocab_size = len(trigrams)
				z = vocab_size - type_count
				if ngram_count == 0:
					prob = float(type_count)/float(z*(prior_count + type_count))
				else:
					prob = float(ngram_count)/float(prior_count + type_count)
				wb_lambda=1-trigrams[w1][w2][w3]/float(trigrams[w1][w2][w3]+sum((trigrams[w1][w2]).values()))
				prob=(wb_lambda)*prob+(1-wb_lambda)*bigrams[w2][w3]/float(unigrams[w2])
				wb_prob.append([w1,w2,w3,prob])
	return wb_prob

def p_kn1(unigrams, bigrams):
    d=10
    #probs_kn=[]
    probs_kn={}
    for w1 in bigrams:
    	for w2 in bigrams[w1]:
    		prob = 0
        	prob = max(bigrams[w1][w2] - d, 0)/float(unigrams[w1])
    		prob += (d/unigrams[w1])*(bigrams[w1][w2]/float(sum(bigrams[w1].values())))
    		if w1 not in probs_kn:
    			probs_kn[w1]={}
    		if w2 not in probs_kn[w1]:
    			probs_kn[w1][w2]=1
    		else:
    			probs_kn[w1][w2]+=1
    		#probs_kn.append([w1,w2,prob])
    return probs_kn

def p_kn2(bigrams,trigrams):
	d=10
	#probs_kn=[]
	probs_kn={}
	for w1 in trigrams:
		for w2 in trigrams[w1]:
			for w3 in trigrams[w1][w2]:
				prob=0
				prob=max(trigrams[w1][w2][w3]-d,0)/float(bigrams[w1][w2])
				prob+=(d/bigrams[w1][w2])*(trigrams[w1][w2][w3]/float(sum(trigrams[w1][w2].values())))
				probs_kn[w1][w2][w3]=prob
				#probs_kn.append([w1,w2,w3,prob])
	return probs_kn

def generate_text_bigrams(unigrams,probs_kn):
	n=randint(0,len(unigrams))
	i=0
	for key1 in unigrams:
		if i==n:
			w1=key1
			break
		i+=1
	i=0
	text=w1
	x=list()
	x.append(w1)
	w2=w1
	for i in xrange(15):
		y=dict(sorted(probs_kn[w1].items(), key=itemgetter(1)))
		#print y
		for w2 in y:
			#print w2
			if w2 in x:
				continue
			else:
				text+=" "+w2
				x.append(w2)
				break
		w1=w2
	return text

def zipf_curve(list):
	x=range(len(list))
	y=sorted(list,key=lambda x: x[-1],reverse=True)
	plt.ylim(0,1)
	plt.plot(x,map(itemgetter(-1),y))

if __name__=="__main__":
	f1=sys.argv[1]
	f2=sys.argv[2]
	f3=sys.argv[3]
	file1 = open(f1)
	file2 = open(f2)
	file3 = open(f3)
	corpus1=file1.read()
	corpus2=file2.read()
	corpus3=file3.read()

	unigrams1=get_unigrams(corpus1)
	bigrams1=get_bigrams(corpus1)
	trigrams1=get_trigrams(corpus1)

	unigrams2=get_unigrams(corpus2)
	bigrams2=get_bigrams(corpus2)
	trigrams2=get_trigrams(corpus2)
	
	unigrams3=get_unigrams(corpus3)
	bigrams3=get_bigrams(corpus3)
	trigrams3=get_trigrams(corpus3)

	##text generation
	# text=generate_text_bigrams(unigrams1,p_kn1(unigrams1,bigrams1))
	# print text

	vocab=0
	for w1 in trigrams1:
		for w2 in trigrams1[w1]:
			vocab+=len(trigrams1[w1][w2])
	zipf_curve(p_trigram(bigrams1,trigrams1))
	zipf_curve(p_trigram(bigrams2,trigrams2))
	zipf_curve(p_trigram(bigrams3,trigrams3))
	#zipf_curve(p_unigram(unigrams1))
	#zipf_curve(laplace_unigrams(unigrams1,200))
	# zipf_curve(p_bigram(unigrams1,bigrams1))
	# zipf_curve(laplace_bigrams(unigrams1,bigrams1,200))
	# zipf_curve(p_kn1(unigrams1,bigrams1))
	# zipf_curve(witten_bell1(unigrams1,bigrams1))
	# zipf_curve(p_trigram(bigrams1,trigrams1))
	# zipf_curve(laplace_trigrams(bigrams1,trigrams1,200))
	# zipf_curve(laplace_trigrams(bigrams1,trigrams1,2000))
	# zipf_curve(laplace_trigrams(bigrams1,trigrams1,vocab))
	# zipf_curve(laplace_trigrams(bigram1s,trigrams1,10*vocab))
	# zipf_curve(p_kn2(bigrams1,trigrams1))
	# zipf_curve(witten_bell2(unigrams1,bigrams1,trigrams1))
	plt.show()
