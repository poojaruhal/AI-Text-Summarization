
import sys, os, re, operator, string
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy
import random
import math
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from nltk.stem import PorterStemmer
from collections import Counter
import sys
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.spatial.distance import pdist, squareform
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


from bllipparser import RerankingParser
import StanfordDependencies

source_path = '/home/admin6019/Downloads/testsentence'



rrp = RerankingParser.fetch_and_load('WSJ-PTB3', verbose=True)
nbest_list = rrp.parse('Why does a zebra have stripes and a giraffe has square spots?')
#questionParsed=rrp.simple_parse('Why does a zebra have stripes and a giraffe has square spots?')
print repr(nbest_list[0])
print nbest_list[0].ptb_parse #parse tree 
print nbest_list[0].parser_score #parser score 
print nbest_list[0].reranker_score # reranker score   
tokens = nbest_list[0].ptb_parse.sd_tokens()
for token in tokens:
     print token


for dirpath, dirs, files in os.walk(source_path):
    for file in files:
       	fname = os.path.join(dirpath, file)
       	print "fname=", fname
	with open(fname) as eachfile:
		text=eachfile.read()
		text = re.sub(r'(M\w{1,2})\.', r'\1', text)
		sentences=re.split(r' *[\.\?!][\'"\)\]]* *', text)
		lowers=[]				
		stemmed=[]
		for sentence in sentences:
			lowers.append(sentence.lower())
						
		for lower in lowers:		
			no_punctuation = lower.translate(None, string.punctuation)
   			tokens = nltk.word_tokenize(no_punctuation)
			#print "generated tokens are :",tokens
			filtered_keywords=[w for w in tokens if not w in stopwords.words('english')]
			answerParse=rrp.parse(filtered_keywords)
			
