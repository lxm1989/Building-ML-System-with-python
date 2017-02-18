from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
content = ["How to format my hard disk","Hard disk format problems"]
X = vectorizer.fit_transform(content)
print(vectorizer.get_feature_names())
print(X.toarray().transpose())