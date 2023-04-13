# Import the logistic regression
from sklearn.linear_model import LogisticRegression

# Define the vector of targets and matrix of features
y = movies.label
X = movies.drop('label', axis=1)

# Build a logistic regression model and calculate the accuracy
log_reg = LogisticRegression().fit(X, y)

print('Accuracy of logistic regression: ', log_reg.score(X, y))
# You have built your first logistic regression model and checked its accuracy! Let's practice some more!
------------
# Define the vector of targets and matrix of features
y = tweets.airline_sentiment
X = tweets.drop('airline_sentiment', axis=1)

# Build a logistic regression model and calculate the accuracy
log_reg = LogisticRegression().fit(X, y)
print('Accuracy of logistic regression: ', log_reg.score(X, y))

# Create an array of prediction
y_predict = log_reg.predict(X)

# Print the accuracy using accuracy score
print('Accuracy of logistic regression: ', accuracy_score(y, y_predict))
# You have built another logistic regression model and calculated its accuracy in two different ways. Have you noticed how the calculated accuracy scores are the same? This will not always be the case for other methods because the .score() function can use other default model performance metrics. So, use accuracy_score() to be certain that you are calculating the accuracy when you are training a different supervised learning model.

-----------

# Import the required packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Define the vector of labels and matrix of features
y = movies.label
X = movies.drop('label', axis=1)

# Perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a logistic regression model and print out the accuracy
log_reg = LogisticRegression().fit(X_train, y_train)
print('Accuracy on train set: ', log_reg.score(X_train, y_train))
print('Accuracy on test set: ', log_reg.score( X_test, y_test))
# Did you notice how the logistic regression's accuracy decreases when we evaluate it on the test set instead of on the training set? It's normal to observe a small drop but if the decrease is large, this could be a signal that your model will not generalize well and will do poorly when evaluating new movie reviews.
#     Accuracy on train set:  0.7861666666666667
#     Accuracy on test set:  0.7521652231845436
----------
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)

# Train a logistic regression
log_reg =LogisticRegression().fit(X_train, y_train)

# Make predictions on the test set
y_predicted = log_reg.predict(X_test)

# Print the performance metrics
print('Accuracy score test set: ',  accuracy_score(y_test, y_predicted))
print('Confusion matrix test set: \n', confusion_matrix(y_test, y_predicted)/len(y_test))
# Although the sentiment category here has 3 classes instead of 2, the way we trained and evaluated the model is the same as with 2 classes. The accuracy on the test data was good and the confusion matrix can also show us which category we are bad at predicting.
#     Accuracy score test set:  0.8031854379977247
#     Confusion matrix test set: 
#      [[0.57337884 0.05346985 0.00568828]
#      [0.04209329 0.13879408 0.02730375]
#      [0.01934016 0.04891923 0.09101251]]
------------

# Import the accuracy and confusion matrix
from sklearn.metrics import confusion_matrix ,accuracy_score

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build a logistic regression
log_reg = LogisticRegression().fit(X_train, y_train)

# Predict the labels 
y_predict = log_reg.predict(X_test)

# Print the performance metrics
print('Accuracy score of test data: ', accuracy_score(y_test, y_predict))
print('Confusion matrix of test data: \n', confusion_matrix(y_test, y_predict)/len(y_test))

# You have successfully built another logistic regression model and evaluated its performance on the test set. Is there any way we can improve the performance of the model? We will discuss that in our next video!
---------

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=321)

# Train a logistic regression
log_reg = LogisticRegression().fit(X_train,y_train)

# Predict the probability of the 0 class
prob_0 = log_reg.predict_proba(X_test)[:, 0]
# Predict the probability of the 1 class
prob_1 = log_reg.predict_proba(X_test)[:, 1]

print("First 10 predicted probabilities of class 0: ", prob_0[:10])
print("First 10 predicted probabilities of class 1: ", prob_1[:10])

# Did you notice how the probabilities of class 0 and class 1 add up to 1 for each instance? In problems where the proportion of one class is larger than the other, we might want to work with predicted probabilities instead of predicted classes

------------
# Split data into training and testing
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=123)

# Train a logistic regression with regularization of 1000
log_reg1 = LogisticRegression(C=1000).fit(X_train, y_train)
# Train a logistic regression with regularization of 0.001
log_reg2 = LogisticRegression(C=0.001).fit(X_train, y_train)

# Print the accuracies
print('Accuracy of model 1: ', log_reg1.score(X_test, y_test))
print('Accuracy of model 2: ', log_reg2.score(X_test, y_test))
# Great work! Did you notice how the model with higher degree of penalization(low C) has lower accuracy than the one with very little penalization(high C)? We often sacrifice some accuracy when we regularize a model but the benefit is lower complexity and lower chance of overfitting.
#     Accuracy of model 1:  0.786
#     Accuracy of model 2:  0.7405
---------

# Build a logistic regression with regularizarion parameter of 100
log_reg1 =  LogisticRegression(C=100).fit(X_train, y_train)
# Build a logistic regression with regularizarion parameter of 0.1
log_reg2 =  LogisticRegression(C= 0.1).fit(X_train, y_train)

# Predict the labels for each model
y_predict1 = log_reg1.predict(X_test)
y_predict2 = log_reg2.predict(X_test)

# Print performance metrics for each model
print('Accuracy of model 1: ', accuracy_score(y_test, y_predict1))
print('Accuracy of model 2: ', accuracy_score(y_test, y_predict2))
print('Confusion matrix of model 1: \n' , confusion_matrix(y_test, y_predict1)/len(y_test))
print('Confusion matrix of model 2: \n', confusion_matrix(y_test, y_predict2)/len(y_test))
# You have trained a more and less flexible logistic regressions to predict the sentiment of tweets and evaluated them using different performance metrics. In this case, we again sacrificed some accuracy when we imposed regularizarion.
------------
# Create and generate a word cloud image
cloud_positives = WordCloud(background_color='white').generate(positive_reviews)
 
# Display the generated wordcloud image
plt.imshow(cloud_positives, interpolation='bilinear') 
plt.axis("off")

# Don't forget to show the final image
plt.show()

--------
# Tokenize each item in the review column
word_tokens = [word_tokenize(review) for review in reviews.review]

# Create an empty list to store the length of the reviews
len_tokens = []

# Iterate over the word_tokens list and determine the length of each item
for i in range(len(word_tokens)):
     len_tokens.append(len(word_tokens[i]))

# Create a new feature for the lengh of each review
reviews['n_words'] = len_tokens 

---------
# Import the TfidfVectorizer and default list of English stop words
from sklearn.feature_extraction.text import TfidfVectorizer,ENGLISH_STOP_WORDS

# Build the vectorizer
vect = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2), max_features=200, token_pattern=r'\b[^\d\W][^\d\W]+\b').fit(reviews.review)
# Create sparse matrix from the vectorizer
X = vect.transform(reviews.review)

# Create a DataFrame
reviews_transformed = pd.DataFrame(X.toarray(), columns=vect.get_feature_names())
print('Top 5 rows of the DataFrame: \n', reviews_transformed.head())
# You have transfomed the text column using the TfidfVectorizer and created 200 numeric columns from the review. You are now ready to build a binary classifier predicting the sentiment of a review.
#      able  action  actually  ago  album  ...  writing  written  wrong  year  years
#     0   0.0     0.0       0.0  0.0    0.0  ...      0.0      0.0    0.0   0.0  0.000
#     1   0.0     0.0       0.0  0.0    0.0  ...      0.0      0.0    0.0   0.0  0.209
#     2   0.0     0.0       0.0  0.0    0.0  ...      0.0      0.0    0.0   0.0  0.152
#     3   0.0     0.0       0.0  0.0    0.0  ...      0.0      0.0    0.0   0.0  0.000
#     4   0.0     0.0       0.0  0.0    0.0  ...      0.0      0.0    0.0   0.0  0.000
-------
# Define X and y
y = reviews_transformed.score
X = reviews_transformed.drop('score', axis=1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=456)

# Train a logistic regression
log_reg = LogisticRegression().fit(X_train, y_train)
# Predict the labels
y_predicted = log_reg.predict(X_test)

# Print accuracy score and confusion matrix on test set
print('Accuracy on the test set: ', accuracy_score(y_test, y_predicted))
print(confusion_matrix(y_test, y_predicted)/len(y_test))

# You have trained and evaluated a logistic regression classifier using product reviews which you have transformed to numeric features. You are now ready to tackle other sentiment analysis problems.
------------


