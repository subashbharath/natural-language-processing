import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
#tokenzation
print("tokenization:Tokenization can be done to either separate words or sentences.")
paragraph = """I have three visions for India. In 3000 years of our history, people from all over 
               the world have come and invaded us, captured our lands, conquered our minds. 
               From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British,
               the French, the Dutch, all of them came and looted us, took over what was ours. 
               Yet we have not done this to any other nation. We have not conquered anyone. 
               We have not grabbed their land, their culture, 
               their history and tried to enforce our way of life on them. 
               Why? Because we respect the freedom of others.That is why my 
               first vision is that of freedom. I believe that India got its first vision of 
               this in 1857, when we started the War of Independence. It is this freedom that
               we must protect and nurture and build on. If we are not free, no one will respect us.
               My second vision for India’s development. For fifty years we have been a developing nation.
               It is time we see ourselves as a developed nation. We are among the top 5 nations of the world
               in terms of GDP. We have a 10 percent growth rate in most areas. Our poverty levels are falling.
               Our achievements are being globally recognised today. Yet we lack the self-confidence to
               see ourselves as a developed nation, self-reliant and self-assured. Isn’t this incorrect?
               I have a third vision. India must stand up to the world. Because I believe that unless India 
               stands up to the world, no one will respect us. Only strength respects strength. We must be 
               strong not only as a military power but also as an economic power. Both must go hand-in-hand. 
               My good fortune was to have worked with three great minds. Dr. Vikram Sarabhai of the Dept. of 
               space, Professor Satish Dhawan, who succeeded him and Dr. Brahm Prakash, father of nuclear material.
               I was lucky to have worked with all three of them closely and consider this the great opportunity of my life. 
               I see four milestones in my career"""
               
# Tokenizing sentences
print("if the text is split into word with one separtion of sentence")
sen_tok=nltk.sent_tokenize(paragraph)
print(sen_tok)

# Tokenizing word
print("if the text is split into word with one separtion")
word_tok=nltk.word_tokenize(paragraph)
print(word_tok)

#stop words
print("""Stop words are those words in the text which does not add any meaning to the sentence and their removal will not affect the processing of text for the defined purpose. 
      They are removed from the vocabulary to reduce noise and to reduce the dimension of the feature set""")
print(stopwords.words("english"))
#stemming
print("Stemming is the process of reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words known as a lemma.")

stemmer = PorterStemmer()
for i in range(len(sen_tok)):
    words=nltk.word_tokenize(sen_tok[i])
    words=[stemmer.stem(word) for word in words if word not in set(stopwords.words("english")) ]
    sen_tok[i]=' '.join(words)
    




