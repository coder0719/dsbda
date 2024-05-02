# %%
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer

# %%
sentence = "Text analytics is the process of deriving meaningful information from natural language text."


# %%
tokens = word_tokenize(sentence)
print(tokens)

# %%
sent_tokens = sent_tokenize(sentence)
print(sent_tokens)

# %%
pos_tags = pos_tag(tokens)
print(pos_tags)

# %%
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print("Unclean version", tokens)
print("\n")
print("Clean version", filtered_tokens)
print("\n")

# %%
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
print("Stemmed words", stemmed_tokens)

# %%
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
print("Lemmatized words are: ", lemmatized)

# %%
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([sentence])
print("\nTerm Frequency and Inverse Document Frequency:")
print(tfidf_matrix.toarray())


