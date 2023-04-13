# "men may come and men may go but i go on forever"?
# (0, 0, 1, 1, 0, 1, 0, 1, 2, 1, 2, 2, 1, 0, 0)
#  the vocabulary has 15 word tokens!
# That is, indeed, the correct answer. Each value in the vector corresponds to the frequency of the corresponding word in the vocabulary.
--------
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create CountVectorizer object
vectorizer = CountVectorizer()

# Generate matrix of word vectors
bow_matrix = vectorizer.fit_transform(corpus)

# Print the shape of bow_matrix
print(bow_matrix.shape)

# output:
#     (7033, 6614)
# You now know how to generate a bag of words representation for a given corpus of documents. Notice that the word vectors created have more than 6600 dimensions. However, most of these dimensions have a value of zero since most words do not occur in a particular tagline.
----------
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create CountVectorizer object
vectorizer = CountVectorizer()

# Generate matrix of word vectors
bow_lem_matrix = vectorizer.fit_transform(lem_corpus)

# Print the shape of bow_lem_matrix
print(bow_lem_matrix.shape)
# output:
#     (6959, 5223)
# Notice how the number of features have reduced significantly from around 6600 to around 5223 for pre-processed movie taglines. The reduced number of dimensions on account of text preprocessing usually leads to better performance when conducting machine learning and it is a good idea to consider it. However, as mentioned in a previous lesson, the final decision always depends on the nature of the application.
------------

# Create CountVectorizer object
vectorizer = CountVectorizer()

# Generate matrix of word vectors
bow_matrix = vectorizer.fit_transform(corpus)

# Convert bow_matrix into a DataFrame
bow_df = pd.DataFrame(bow_matrix.toarray())

# Map the column names to vocabulary 
bow_df.columns = vectorizer.get_feature_names()

# Print bow_df
print(bow_df)

# Observe that the column names refer to the token whose frequency is being recorded. Therefore, since the first column name is an, the first feature represents the number of times the word 'an' occurs in a particular sentence. get_feature_names() essentially gives us a list which represents the mapping of the feature indices to the feature name in the vocabulary.
-----------

# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create a CountVectorizer object
vectorizer = CountVectorizer(lowercase=True, stop_words='english')

# Fit and transform X_train
X_train_bow = vectorizer.fit_transform(X_train)

# Transform X_test
X_test_bow = vectorizer.transform(X_test)

# Print shape of X_train_bow and X_test_bow
print(X_train_bow.shape)
print(X_test_bow.shape)

# output:
#     (250, 8158)
#     (750, 8158)
# You now have a good idea of preprocessing text and transforming them into their bag-of-words representation using CountVectorizer. In this exercise, you have set the lowercase argument to True. However, note that this is the default value of lowercase and passing it explicitly is not necessary. Also, note that both X_train_bow and X_test_bow have 8158 features. There were words present in X_test that were not in X_train. CountVectorizer chose to ignore them in order to ensure that the dimensions of both sets remain the same.
-------------

# Create a MultinomialNB object
clf = MultinomialNB()

# Fit the classifier
clf.fit(X_train_bow, y_train)

# Measure the accuracy
accuracy = clf.score(X_test_bow, y_test)
print("The accuracy of the classifier on the test set is %.3f" % accuracy)

# Predict the sentiment of a negative review
review = "The movie was terrible. The music was underwhelming and the acting mediocre."
prediction = clf.predict(vectorizer.transform([review]))[0]
print("The sentiment predicted by the classifier is %i" % (prediction))

# output:
#     The accuracy of the classifier on the test set is 0.732
#     The sentiment predicted by the classifier is 0
# You have successfully performed basic sentiment analysis. Note that the accuracy of the classifier is 73.2%. Considering the fact that it was trained on only 750 reviews, this is reasonably good performance. The classifier also correctly predicts the sentiment of a mini negative review which we passed into it.

--------

# Generate n-grams upto n=1
vectorizer_ng1 = CountVectorizer(ngram_range=(1,1))
ng1 = vectorizer_ng1.fit_transform(corpus)

# Generate n-grams upto n=2
vectorizer_ng2 = CountVectorizer(ngram_range=(1,2))
ng2 = vectorizer_ng2.fit_transform(corpus)

# Generate n-grams upto n=3
vectorizer_ng3 = CountVectorizer(ngram_range=(1, 3))
ng3 = vectorizer_ng3.fit_transform(corpus)

# Print the number of features for each model
print("ng1, ng2 and ng3 have %i, %i and %i features respectively" % (ng1.shape[1], ng2.shape[1], ng3.shape[1]))
# output:
#     ng1, ng2 and ng3 have 6614, 37100 and 76881 features respectively
# You now know how to generate n-gram models containing higher order n-grams. Notice that ng2 has over 37,000 features whereas ng3 has over 76,000 features. This is much greater than the 6,000 dimensions obtained for ng1. As the n-gram range increases, so does the number of features, leading to increased computational costs and a problem known as the curse of dimensionality.
---------
# Define an instance of MultinomialNB 
clf_ng = MultinomialNB()

# Fit the classifier 
clf_ng.fit(X_train_ng , y_train)

# Measure the accuracy 
accuracy = clf_ng.score(X_test_ng , y_test)
print("The accuracy of the classifier on the test set is %.3f" % accuracy)

# Predict the sentiment of a negative review
review = "The movie was not good. The plot had several holes and the acting lacked panache."
prediction = clf_ng.predict(ng_vectorizer.transform([review]))[0]
print("The sentiment predicted by the classifier is %i" % (prediction))

# output:
#     The accuracy of the classifier on the test set is 0.758
#     The sentiment predicted by the classifier is 0
#  You're now adept at performing sentiment analysis using text. Notice how this classifier performs slightly better than the BoW version. Also, it succeeds at correctly identifying the sentiment of the mini-review as negative. In the next chapter, we will learn more complex methods of vectorizing textual data.
---------------
start_time = time.time()
# Splitting the data into training and test sets
train_X, test_X, train_y, test_y = train_test_split(df['review'], df['sentiment'], test_size=0.5, random_state=42, stratify=df['sentiment'])

# Generating ngrams
vectorizer = CountVectorizer(ngram_range=(1, 3))
train_X = vectorizer.fit_transform(train_X)
test_X = vectorizer.transform(test_X)

# Fit classifier
clf = MultinomialNB()
clf.fit(train_X, train_y)

# Print accuracy, time and number of dimensions
print("The program took %.3f seconds to complete. The accuracy on the test set is %.2f. The ngram representation had %i features." % (time.time() - start_time, clf.score(test_X, test_y), train_X.shape[1]))
# The program took around 0.2 seconds in the case of the unigram model and more than 10 times longer for the higher order n-gram model. The unigram model had over 12,000 features whereas the n-gram model for upto n=3 had over 178,000! Despite taking higher computation time and generating more features, the classifier only performs marginally better in the latter case, producing an accuracy of 77% in comparison to the 75% for the unigram model.
----------
