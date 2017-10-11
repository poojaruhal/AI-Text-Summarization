
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

#question parsing
question="Why does a zebra have stripes and a giraffe has square spots?" #query string
question_vector = CountVectorizer(stop_words='english') #initilizing question vector
question_matrix =  question_vector.fit_transform([question]) #conveting to vector

source_path = 'testsentence' #source directory
preprocessed_dict={} 
question_preproc=[]
finalList=[]
stemmer = PorterStemmer() #initializaton nltk stemmer 

WORD = re.compile(r'\w+')


def calculate_cosine_sim(input_sentence, g, method='cosine'): #method for calculating cosine similarity between each sentences
    final_score=0.0
    data = TextBlob(input_sentence)# taking input as text blob
    if g < 1:
        g = int(round(g*len(data.sentences)))

   
   
    cosine_scores = []
    if method == 'sum':

        sentence_vector = TfidfVectorizer(stop_words='english')#initializing TFIDF vector
        data_matrix = sentence_vector.fit_transform((str(x) for x in data.sentences)) #converting to vectors
          
    else:
        # create count matrix for distance method
        sentence_vector = CountVectorizer(stop_words='english') #initializing count vector
        data_matrix =  sentence_vector.fit_transform((str(x) for x in data.sentences)) #converting to vectors
        # calculate pair-wise distances between each sentence
        data_matrix = data_matrix.toarray()
        y = pdist(data_matrix, method)# calculating cosine distance betwwen every two statements
        
        for var in y:
                cosine_scores.append((-1*var)+1) # convering to cosine similarity
                final_score=final_score+((-1*var)+1)
        

    return final_score,cosine_scores


#reading input directory for input files
for dirpath, dirs, files in os.walk(source_path):
    with open( 'testsentence/result.txt', 'w' ) as result:    
        for file in files:
                fname = os.path.join(dirpath, file)
                print "fname=", fname
                with open(fname) as f:
                        result.write(f.read())
                        


with open('testsentence/result.txt') as merged_file:
    input_file = merged_file.read()
    print "read"
 
final_sim_score,cos_scores = calculate_cosine_sim(input_file, 1) #calculating cosine similarities on input
print "summarize done"


sentence_count=0 # no. of sentences in document cluster

#stemming
def stem_tokens(token, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

with open('testsentence/result.txt') as eachfile:
        text=eachfile.read()
        text = re.sub(r'(M\w{1,2})\.', r'\1', text)
	#splitting files into sentences
        sentences=re.split(r' *[\.\?!][\'"\)\]]* *', text) 
        lowers=[]				
        stemmed=[]
        sentences.pop()	
        for sentence in sentences:
                lowers.append(sentence.lower()) # converting to lower case
                sentence_count=sentence_count+1
        for lower in lowers:		
                no_punctuation = lower.translate(None, string.punctuation)
                tokens = nltk.word_tokenize(no_punctuation)
                filtered_keywords=[w for w in tokens if not w in stopwords.words('english')]
                count = Counter(filtered_keywords)
                preprocessed_dict=dict((str(k), v) for k, v in count.iteritems())
		#creating statements dictionary
                finalList.append(preprocessed_dict)
        #question preprocessing              
        qstemmed=[]
        questionLower=question.lower()
        qno_punctuation = questionLower.translate(None, string.punctuation)
        qtokens = nltk.word_tokenize(qno_punctuation)
        qfiltered_keywords=[w for w in qtokens if not w in stopwords.words('english')]
        qstemmed=stem_tokens(qfiltered_keywords, stemmer)
        qcount = Counter(qfiltered_keywords)
        question_preproc=dict((str(k), v) for k, v in qcount.iteritems())
        
        temp=[]
	dictlist=[]
        cosine_score=[[]]
        for w in finalList:
                for key, value in w.iteritems():
                        temp = [key,value]
                        dictlist.append(temp)

        i=0
        print "total sentence count is ",sentence_count
        relsq=[]
        finalSum = 0.0
	#calculating rel(s|q) vetween sentences and questions	
        for sentence in sentences:
                print " sentence : ",i+1 , "   '\n'"
                sum=0
                
                for w in qfiltered_keywords:
                        tfwq=0.0
                        tfws=0.0
                        sfw=0.0
                
                                                        
                        for word,freq in question_preproc.iteritems():
                                if word == w:
                                        tfwq = freq
                        for word,freq2 in finalList[i].iteritems():
                                if word == w:
                                        tfws =freq2
                        ct=0

                        for j in range(0,sentence_count):				
                                for word,freq in finalList[j].iteritems():
                                        if word == w:
                                                ct=ct+1
                        sfw=ct
                        
                        sum =sum + ( (math.log(tfws+1) )*( math.log(tfwq+1) )*(  math.log(float(sentence_count+1)/(0.5+sfw))  )  )
                relsq.append(sum)
                i=i+1
                finalSum = finalSum +sum

        psq= []	
        d=0.85 #bias value
        inc=0
        final_cos_scores=[]
        start=0
        end=sentence_count-1
        while end > 0:
                final_cos_scores.append(np.sum(cos_scores[start:(start+end)]))
                start=start+end
                end=end-1	
        final_cos_scores.append(0.0)
        for sentence in sentences:
                psq.append(d*(relsq[inc]/finalSum) + (1-d)* (final_cos_scores[inc]/final_sim_score)  )
                inc=inc+1;

        
        
        sorted_final_cos_scores=[i[0] for i in sorted(enumerate(psq), key=lambda x:x[1],reverse=True)]

        top_sentences = 6 # no. of sentences to be output out
	#generating outputs and printing in output file
        with open( 'testsentence/summary.txt', 'w' ) as answer:
		for x in range(0,top_sentences):
                	print sentences[sorted_final_cos_scores[x]]
			answer.write(sentences[sorted_final_cos_scores[x]])                
                                        
sys.exit(0) 


