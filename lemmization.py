import nltk
from nltk.stem import LancasterStemmer
Lanc_stemmer = LancasterStemmer()

word_data = "UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS"
nltk_tokens = nltk.word_tokenize(word_data)
for w in nltk_tokens:
       print("Actual: %s  Stem: %s"  % (w,Lanc_stemmer.stem(w)))