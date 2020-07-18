import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
df=pd.read_csv("smsdata",sep="\t",names=["label","message"])
corpus=[]
#cleaning the data
#stemming
stemmer = PorterStemmer()
print("Stemming is the process of reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words known as a lemma.")

for i in range(0,len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df["message"][i])
    review = review.lower()
    review = review.split()#same as word_tokenize
    review = [stemmer.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    

#bag of words using countvector
# Creating the Bag of Words model
print("Transforms text into a sparse matrix of n-gram counts")
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()

y=pd.get_dummies(df["label"],drop_first=True)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()

#fitting the model
nb.fit(x_train,y_train)
y_pred=nb.predict(x_test)


#performance metrices
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)


#finding the accuarcy score
from sklearn.metrics import accuracy_score
acc_score=accuracy_score(y_test,y_pred)
print(acc_score)