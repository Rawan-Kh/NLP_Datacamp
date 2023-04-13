# Find the number of positive and negative reviews
print('Number of positive and negative reviews: ', movies.label.value_counts())

# Find the proportion of positive and negative reviews
print('Proportion of positive and negative reviews: ', movies.label.value_counts() / len(movies))

#  The .value_counts() method is an easy way to gain a first impression about the contents of the label column.
-----------
length_reviews = movies.review.str.len()

# How long is the shortest review
print(max(length_reviews))

-----------
length_reviews = movies.review.str.len()

# How long is the shortest review
print(min(length_reviews))

# Not only did you gain an idea about your reviews but this approach can be applied to perform other operations on character columns.
---------
# Import the required packages
from textblob import TextBlob

# Create a textblob object  
blob_two_cities = TextBlob(two_cities)

# Print out the sentiment 
print(blob_two_cities.sentiment)

# Looking at the string, do you agree with its overall slightly positive score?
# Sentiment(polarity=0.022916666666666658, subjectivity=0.5895833333333332)
------
# Import the required packages
from textblob import TextBlob

# Create a textblob object 
blob_annak = TextBlob(annak)
blob_catcher = TextBlob(catcher)

# Print out the sentiment   
print('Sentiment of annak: ', blob_annak.sentiment)
print('Sentiment of catcher: ', blob_catcher.sentiment)

# t shouldn't be surprising that the opening sentence of _Catcher in the Rye_ has a negative score, whereas the one from _Anna Karenina_ has a slightly positive one.
# Sentiment of annak:  Sentiment(polarity=0.05000000000000002, subjectivity=0.95)
# Sentiment of catcher:  Sentiment(polarity=-0.05, subjectivity=0.5466666666666666)
------------

# Import the required packages
from textblob import TextBlob

# Create a textblob object  
blob_titanic = TextBlob(titanic)

# Print out its sentiment  
print(blob_titanic.sentiment)

# Did you notice that the polarity is around 0.2 and the review is classified as positive (has a label of 1)?
# Sentiment(polarity=0.2024748060772906, subjectivity=0.4518248900857597)

-------------
from wordcloud import WordCloud

# Generate the word cloud from the east_of_eden string
cloud_east_of_eden = WordCloud(background_color="white").generate(east_of_eden)

# Create a figure of the generated cloud
plt.imshow(cloud_east_of_eden, interpolation='bilinear')  
plt.axis('off')
# Display the figure
plt.show()
-------------

# Import the word cloud function  
from wordcloud import WordCloud

# Create and generate a word cloud image 
my_cloud = WordCloud(background_color='white', stopwords=my_stopwords).generate(descriptions)

# Display the generated wordcloud image
plt.imshow(my_cloud, interpolation='bilinear') 
plt.axis("off")

# Don't forget to show the final image
plt.show()
-----------



