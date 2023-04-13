# Bag-of-words is a simple but effective method to build a vocabulary of all the words occurring in a document.
# You'll next see how to apply this idea to sentiment analysis further
-------
# Import the required function
from sklearn.feature_extraction.text import CountVectorizer

annak = ['Happy families are all alike;', 'every unhappy family is unhappy in its own way']

# Build the vectorizer and fit it
anna_vect = CountVectorizer()
anna_vect.fit(annak)

# Create the bow representation
anna_bow = anna_vect.transform(annak)

# Print the bag-of-words result 
print(anna_bow.toarray())

# You have transformed the first sentence of _Anna Karenina_ to an array counting the frequencies of each word. However, the output is not very readable, is it? We are still missing the names of the features. And does the approach change when we apply it to a larger dataset? We explore these problems in the next exercise.
#  [[1 1 1 0 1 0 1 0 0 0 0 0 0]
#  [0 0 0 1 0 1 0 1 1 1 1 2 1]]
---------

from sklearn.feature_extraction.text import CountVectorizer 

# Build the vectorizer, specify max features 
vect = CountVectorizer(max_features=100)
# Fit the vectorizer
vect.fit(reviews.review)

# Transform the review column
X_review = vect.transform(reviews.review)

# Create the bow representation
X_df=pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())
print(X_df.head())
# You have successfully built your first BOW generated vocabulary and transformed it to numeric features of the dataset!
--------

from sklearn.feature_extraction.text import CountVectorizer 

# Build the vectorizer, specify token sequence and fit
vect = CountVectorizer(ngram_range=(1,2))
vect.fit(reviews.review)

# Transform the review column
X_review = vect.transform(reviews.review)

# Create the bow representation
X_df = pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())
print(X_df.head())
# You have built a numeric representation of the review column using uni- and bigrams!
-------
# Using the movies dataset, limit the size of the vocabulary to 100.
from sklearn.feature_extraction.text import CountVectorizer 

# Build the vectorizer, specify size of vocabulary and fit
vect = CountVectorizer(max_features=100)
vect.fit(movies.review)

# Transform the review column
X_review = vect.transform(movies.review)
# Create the bow representation
X_df = pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())
print(X_df.head())
-----------

# Using the movies dataset, limit the size of the vocabulary to include terms which occur in no more than 200 documents.
from sklearn.feature_extraction.text import CountVectorizer 

# Build and fit the vectorizer
vect = CountVectorizer(max_df=200)
vect.fit(movies.review)

# Transform the review column
X_review = vect.transform(movies.review)
# Create the bow representation
X_df = pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())
print(X_df.head())

------
# Using the movies dataset, limit the size of the vocabulary to ignore terms which occur in less than 50 documents.
from sklearn.feature_extraction.text import CountVectorizer 

# Build and fit the vectorizer
vect = CountVectorizer(min_df=50)
vect.fit(movies.review)

# Transform the review column
X_review = vect.transform(movies.review)
# Create the bow representation
X_df = pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())
print(X_df.head())

# Any of the three methods you applied here can be used to limit the size of the vocabulary. Which of the three methods you used resulted in the lowest number of constructed features?
-------

#Import the vectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Build the vectorizer, specify max features and fit
vect = CountVectorizer(max_features=1000, ngram_range=(2, 2), max_df=500)
vect.fit(reviews.review)

# Transform the review
X_review = vect.transform(reviews.review)

# Create a DataFrame from the bow representation
X_df = pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())
print(X_df.head())

# You have successfully created a bag-of-words representation of the product reviews dataset, including more sophisticated sequence of tokens, while limiting the size of the vocabulary
---------

# Import the required function
from nltk import word_tokenize

# Transform the GoT string to word tokens
print(word_tokenize(GoT))

# You have successfully taken a string and split it up into word tokens.
-----
# Import the word tokenizing function
from nltk import word_tokenize

# Tokenize each item in the avengers 
tokens_avengers = [word_tokenize(item) for item in avengers]

print(tokens_avengers)
#  You have built up on what you developed in the previous exercise and created a list comprehension where each of the items in the list is a quote from an Avengers movie.
--------

# Import the needed packages
from nltk import word_tokenize

# Tokenize each item in the review column 
word_tokens = [word_tokenize(review) for review in reviews.review]

# Print out the first item of the word_tokens list
print(word_tokens[0])

-------
# Create an empty list to store the length of the reviews
len_tokens = []

# Iterate over the word_tokens list and determine the length of each item
for i in range(len(word_tokens)):
     len_tokens.append(len(word_tokens[i]))

# Create a new feature for the lengh of each review
reviews['n_words'] = len_tokens 
# You have used a list comprehension and a for loop to iterate over the word tokens created from the review column. You can employ the same approach to create other features, such as one counting the number of sentences in each review. This knowledge will also help you understand the next chapter.
---------

# Import the language detection function and package
from langdetect import detect_langs

# Detect the language of the foreign string
print(detect_langs(foreign))
# ou have successfully identified the language of the string to be French!
----------
from langdetect import detect_langs

languages = []

# Loop over the sentences in the list and detect their language
for sentence in sentences:
    languages.append(detect_langs(sentence))
    
print('The detected languages are: ', languages)

--------
from langdetect import detect_langs
languages = [] 

# Loop over the rows of the dataset and append  
for row in range(len(non_english_reviews)):
    languages.append(detect_langs(non_english_reviews.iloc[row, 1]))

# Clean the list by splitting     
languages = [str(lang).split(':')[0][1:] for lang in languages]

# Assign the list to a new feature 
non_english_reviews['language'] = languages

print(non_english_reviews.head())

# You have succesfully built a new column in the dataset, which tells you in which language the respective review is written. This can be a very useful feature!
----------
# Import the word cloud function 
from wordcloud  import WordCloud 

# Create and generate a word cloud image
my_cloud = WordCloud(background_color='white').generate(text_tweet)

# Display the generated wordcloud image
plt.imshow(my_cloud, interpolation='bilinear') 
plt.axis("off")

# Don't forget to show the final image
plt.show()

---------
# Import the word cloud function and stop words list
from wordcloud import WordCloud, STOPWORDS 

# Define and update the list of stopwords
my_stop_words = STOPWORDS.update(['airline', 'airplane'])

# Create and generate a word cloud image
my_cloud = WordCloud(stopwords=my_stop_words).generate(text_tweet)

# Display the generated wordcloud image
plt.imshow(my_cloud, interpolation='bilinear') 
plt.axis("off")
# Don't forget to show the final image
plt.show()
# Do you notice any changes in the first word cloud where you did not remove the stop words and the second one, where you removed them? If the change is not so obvious, perhaps the list of stop words needs to be enriched further.
-------------
# Import the stop words
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# Define the stop words
my_stop_words = ENGLISH_STOP_WORDS.union(['airline', 'airlines', '@'])

# Build and fit the vectorizer
vect = CountVectorizer(stop_words=my_stop_words)
vect.fit(tweets.text)

# Create the bow representation
X_review = vect.transform(tweets.text)
# Create the data frame
X_df = pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())
print(X_df.head())
# Did you notice that in this case the created features contain digits and other characters? Social media data can in general be quite messy and in a later video we will learn how to remove all digits and other characters and retain only more meaningful features.
#   00  000  000114  000419  0011  ...  zero  zfqmpgxvs6  zone  zsuztnaijq  zv2pt6trk9
#     0   0    0       0       0     0  ...     0           0     0           0           0
#     1   0    0       0       0     0  ...     0           0     0           0           0
#     2   0    0       0       0     0  ...     0           0     0           0           0
#     3   0    0       0       0     0  ...     0           0     0           0           0
#     4   0    0       0       0     0  ...     0           0     0           0           0
---------

# Import the vectorizer and default English stop words list
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# Define the stop words
my_stop_words = ENGLISH_STOP_WORDS.union(['airline', 'airlines', '@', 'am', 'pm'])
 
# Build and fit the vectorizers
vect1 = CountVectorizer(stop_words=my_stop_words)
vect2 = CountVectorizer(stop_words=ENGLISH_STOP_WORDS) 
vect1.fit(tweets.text)
vect2.fit(tweets.negative_reason)

# Print the last 15 features from the first, and all from second vectorizer
print(vect1.get_feature_names()[-15:])
print(vect2.get_feature_names())

# We can have multiple text columns in a single dataset. In that case, we can transform each of them to numeric features separately, using different arguments in the CountVectorizer() function.
-----------

# Build and fit the vectorizer
vect = CountVectorizer(token_pattern=r'\b[^\d\W][^\d\W]+\b').fit(tweets.text)
vect.transform(tweets.text)
print('Length of vectorizer: ', len(vect.get_feature_names()))

----------

# Build the first vectorizer
vect1 = CountVectorizer(token_pattern=r'\b[^\d\W][^\d\W]').fit(tweets.text)
vect1.transform(tweets.text)

# Build the second vectorizer
vect2 = CountVectorizer(token_pattern=r'\b[^\d\W][^\d\W]').fit(tweets.text)
vect2.transform(tweets.text)

# Print out the length of each vectorizer
print('Length of vectorizer 1: ', len(vect1.get_feature_names()))
print('Length of vectorizer 2: ', len(vect2.get_feature_names()))
# Did you notice how fewer features were created when we specified the token pattern? It is a nice way to limit the size of our vocabulary and make sure we only include certain tokens when we create it.
---------------

# Import the word tokenizing package
from nltk import word_tokenize

# Tokenize the text column
word_tokens = [word_tokenize(review) for review in tweets.text]
print('Original tokens: ', word_tokens[0])

# Filter out non-letter characters
cleaned_tokens = [[word for word in item if word.isalpha()] for item in word_tokens]
print('Cleaned tokens: ', cleaned_tokens[0])
# Did you notice how the list of word tokens changes before and after the filtering out of non-alphabetic characters
---------

# Create a list of lists, containing the tokens from list_tweets
tokens = [word_tokenize(item) for item in tweets_list]

# Remove characters and digits , i.e. retain only letters
letters = [[word for word in item if word.isalpha()] for item in tokens]
# Remove characters, i.e. retain only letters and digits
let_digits = [[word for word in item if word.isalnum()] for item in tokens]
# Remove letters and characters, retain only digits
digits = [[word for word in item if word.isdigit()] for item in tokens]

# Print the last item in each list
print('Last item in alphabetic list: ', letters[2])
print('Last item in list of alphanumerics: ', let_digits[2])
print('Last item in the list of digits: ', digits[2])
# You now know how to apply string operators to modify strings or lists of strings. You can apply these skills when constructing features from text.
-----

# Import the required packages from nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import word_tokenize

porter = PorterStemmer()
WNlemmatizer = WordNetLemmatizer()

# Tokenize the GoT string
tokens = word_tokenize(GoT) 

----
import time

# Log the start time
start_time = time.time()

# Build a stemmed list
stemmed_tokens = [porter.stem(token) for token in tokens] 

# Log the end time
end_time = time.time()

print('Time taken for stemming in seconds: ', end_time - start_time)
print('Stemmed tokens: ', stemmed_tokens) 

-----------
import time

# Log the start time
start_time = time.time()

# Build a lemmatized list
lem_tokens = [WNlemmatizer.lemmatize(token) for token in tokens]

# Log the end time
end_time = time.time()

print('Time taken for lemmatizing in seconds: ', end_time - start_time)
print('Lemmatized tokens: ', lem_tokens) 


# You can use stemming or lemmatization to transform lists of tokens. Which one to choose will depend on the problem. Did you notice how much longer lemmatization takes compared to stemming?
---------

# Import the language detection package
import langdetect 

# Loop over the rows of the dataset and append  
languages = [] 
for i in range(len(non_english_reviews)):
    languages.append(langdetect.detect_langs(non_english_reviews.iloc[i, 1]))

# Clean the list by splitting     
languages = [str(lang).split(':')[0][1:] for lang in languages]
# Assign the list to a new feature 
non_english_reviews['language'] = languages

# Select the Spanish ones
filtered_reviews = non_english_reviews[non_english_reviews.language == 'es']

-----------
# Import the required packages
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize

# Import the Spanish SnowballStemmer
SpanishStemmer = SnowballStemmer("spanish")

# Create a list of tokens
tokens = [word_tokenize(review) for review in filtered_reviews.review]
 
# Stem the list of tokens
stemmed_tokens = [[SpanishStemmer.stem(word) for word in token] for token in tokens]

# Print the first item of the stemmed tokenss
print(stemmed_tokens[0])
# You have combined bits and pieces you have learned throughout the course to detect the reviews which are in Spanish and created a list of stemmed tokens from them.
---------

# Import the function to perform stemming
from nltk.stem import PorterStemmer
from nltk import word_tokenize

# Call the stemmer
porter = PorterStemmer()

# Transform the array of tweets to tokens
tokens = [word_tokenize(tweet) for tweet in tweets]
# Stem the list of tokens
stemmed_tokens = [[porter.stem(word) for word in tweet] for tweet in tokens] 
# Print the first element of the list
print(stemmed_tokens[0])
# You have created your own list of tokens and turned them into stems! Are there other ways we can still improve the output of our tokenization and numerical representation from text? In the next lesson, we will learn a new method!
------

# Import the required function
from sklearn.feature_extraction.text import TfidfVectorizer

annak = ['Happy families are all alike;', 'every unhappy family is unhappy in its own way']

# Call the vectorizer and fit it
anna_vect = TfidfVectorizer().fit(annak)

# Create the tfidf representation
anna_tfidf = anna_vect.transform(annak)

# Print the result 
print(anna_tfidf.toarray())
# You have built your first numeric representation of text by applying a TfIdf vectorizer. Do you recall building a bag-of-words representation for the same data earlier? What differences do you notice?
------------
# Import the required vectorizer package and stop words list
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# Define the vectorizer and specify the arguments
my_pattern = r'\b[^\d\W][^\d\W]+\b'
vect = TfidfVectorizer(ngram_range=(1, 2), max_features=100, token_pattern=my_pattern, stop_words=ENGLISH_STOP_WORDS).fit(tweets.text)

# Transform the vectorizer
X_txt = vect.transform(tweets.text)

# Transform to a data frame and specify the column names
X=pd.DataFrame(X_txt.toarray(), columns=vect.get_feature_names())
print('Top 5 rows of the DataFrame: ', X.head())
# You now can succesfully apply two different methods to transform a text column of any kind to a numeric form. We need to implement this step in order to apply a supervised machine learning model to a sentiment analysis problem.
# Top 5 rows of the DataFrame:     agent  airline  airport    amp  austin  ...  wait  way  website  work  yes
#                               0    0.0      0.0      0.0  0.000     0.0  ...   0.0  0.0      0.0   0.0  0.0
#                               1    0.0      0.0      0.0  0.000     0.0  ...   0.0  0.0      0.0   0.0  0.0
#                               2    0.0      0.0      0.0  0.000     0.0  ...   0.0  0.0      0.0   0.0  0.0
#                               3    0.0      0.0      0.0  0.634     0.0  ...   0.0  0.0      0.0   0.0  0.0
#                               4    0.0      0.0      0.0  0.000     0.0  ...   0.0  0.0      0.0   0.0  0.0
---------

# Import the required packages
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

# Build a BOW and tfidf vectorizers from the review column and with max of 100 features
vect1 = CountVectorizer(max_features=100).fit(reviews.review)
vect2 = TfidfVectorizer(max_features=100).fit(reviews.review) 

# Transform the vectorizers
X1 = vect1.transform(reviews.review)
X2 = vect2.transform(reviews.review)
# Create DataFrames from the vectorizers 
X_df1 = pd.DataFrame(X1.toarray(), columns=vect1.get_feature_names())
X_df2 = pd.DataFrame(X2.toarray(), columns=vect2.get_feature_names())
print('Top 5 rows using BOW: \n', X_df1.head())
print('Top 5 rows using tfidf: \n', X_df2.head())

# You can now successfully transform text features into numeric ones using two different approaches. Which approach should you select? That usually depends on the context and on how well they perform when used with a machine learning model.
-------------















