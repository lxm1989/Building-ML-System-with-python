import os

posts = [open(os.path.join("data/",f)).read() for f in os.listdir("data")]

#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk.stem
english_stemmer=nltk.stem.SnowballStemmer('english')

#class StemmedCountVec(CountVectorizer):
#    def build_analyzer(self):
#        analyzer=super(StemmedCountVec,self).build_analyzer()
#        return lambda doc:(english_stemmer.stem(w) for w in analyzer(doc))

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer=super(StemmedTfidfVectorizer,self).build_analyzer()
        return lambda doc:(english_stemmer.stem(w) for w in analyzer(doc))

#vectorizer = CountVectorizer(stop_words='english')
#vectorizer = StemmedCountVec(stop_words='english')
vectorizer = StemmedTfidfVectorizer(stop_words='english',decode_error ='ignore')
X_train = vectorizer.fit_transform(posts)
print(vectorizer.get_feature_names())
new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])

import scipy as sp
##import linalg from scipy
def dist_raw(v1,v2):
    v1_norm = v1/sp.linalg.norm(v1.toarray())
    v2_norm = v2/sp.linalg.norm(v2.toarray())
    delta = v1_norm - v2_norm
    return sp.linalg.norm(delta.toarray())

import sys
best_doc = None
best_dist = sys.maxsize
best_i = None
for i in range(0,len(posts)):
    post=posts[i]
    if post==new_post:
        continue
    post_vec=X_train.getrow(i)
    d=dist_raw(post_vec,new_post_vec)
    print("=== Post %i with dist=%.2f: %s"%(i,d,post))
    if d < best_dist:
        best_dist = d
        best_i = i
print("Best post is %i with dist=%.2f"%(best_i,best_dist))


