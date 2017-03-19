import sklearn.datasets

#sklearn.datasets.fetch_20newsgroups(data_home=None, subset='train', categories=None, shuffle=True, random_state=42, remove=(), download_if_missing=True)
groups = [
    'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware', 'comp.windows.x', 'sci.space']

train_data = sklearn.datasets.fetch_20newsgroups(categories=groups,subset="train",data_home="~/Documents/Building ML System with python/chat3")
#print(len(train_data.filenames))

#test_data = sklearn.datasets.fetch_20newsgroups(categories=groups,subset="test",data_home="~/Documents/Building ML System with python/chat3")
#print(len(test_data.filenames))


from sklearn.feature_extraction.text import TfidfVectorizer
import nltk.stem
english_stemmer=nltk.stem.SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer=super(StemmedTfidfVectorizer,self).build_analyzer()
        return lambda doc:(english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedTfidfVectorizer(min_df=10,max_df=0.5,stop_words='english',decode_error='ignore')
vectorized = vectorizer.fit_transform(train_data.data)
num_sample,num_features = vectorized.shape
#print(num_sample,"::",num_features)
num_clusters = 50
from sklearn.cluster import KMeans
km = KMeans(n_clusters=num_clusters,init='random',n_init=1,verbose=1)
km.fit(vectorized)
#km.labels_
#km.cluster_centers_
new_post = \
    """Disk drive problems. Hi, I have a problem with my hard disk.
After 1 year it is working only sporadically now.
I tried to format it, but now it doesn't boot any more.
Any ideas? Thanks.
"""
new_post_vec = vectorizer.transform([new_post])
new_post_label = km.predict(new_post_vec)[0]
similar_indices = (km.labels_==new_post_label).nonzero()[0]


import scipy as sp
similar = []
for i in similar_indices:
    dist = sp.linalg.norm((new_post_vec-vectorized[i]).toarray())
    similar.append((dist,train_data.data[i]))

similar = sorted(similar)

show_at_1 = similar[0]
show_at_2 = similar[int(len(similar) / 10)]
show_at_3 = similar[int(len(similar) / 2)]

print("=== #1 ===")
print(show_at_1)
print()

print("=== #2 ===")
print(show_at_2)
print()

print("=== #3 ===")
print(show_at_3)



post_group = zip(train_data.data,train_data.target)
z = [(len(post[0]),post[0],train_data.target_names[post[1]]) for post in post_group]
analyzer = vectorizer.build_analyzer()
list(set(analyzer(z[5][1])).intersection(vectorizer.get_feature_names()))