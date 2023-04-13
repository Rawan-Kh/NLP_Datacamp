In fact, the tf-idf weight for bottle in every document will be 0. This is because the inverse document frequency is constant across documents in a corpus and since bottle occurs in every document, its value is log(1), which is 0
--------
# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TfidfVectorizer object
vectorizer= TfidfVectorizer()

# Generate matrix of word vectors
tfidf_matrix = vectorizer.fit_transform(ted)

# Print the shape of tfidf_matrix
print(tfidf_matrix.shape)
# You now know how to generate tf-idf vectors for a given corpus of text. You can use these vectors to perform predictive modeling just like we did with CountVectorizer. In the next few lessons, we will see another extremely useful application of the vectorized form of documents: generating recommendations.
---------
# Since document vectors use only non-negative weights, the cosine score lies between 0 and 1.
---------
# Initialize numpy vectors
A = np.array([1,3])
B = np.array([-2,2])

# Compute dot product
dot_prod = np.dot(A, B)

# Print dot product
print(dot_prod)
# Good job! The dot product of the two vectors is 1 * -2 + 3 * 2 = 4, which is indeed the output produced. We will not be using np.dot() too much in this course but it can prove to be a helpful function while computing dot products between two standalone vectors.
----------

# Initialize an instance of tf-idf Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Generate the tf-idf vectors for the corpus
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Compute and print the cosine similarity matrix
cosine_sim = cosine_similarity( tfidf_matrix,tfidf_matrix)
print(cosine_sim)
# As you will see in a subsequent lesson, computing the cosine similarity matrix lies at the heart of many practical systems such as recommenders. From our similarity matrix, we see that the first and the second sentence are the most similar. Also the fifth sentence has, on average, the lowest pairwise cosine scores. This is intuitive as it contains entities that are not present in the other sentences.
# output:
#     [[1.         0.36413198 0.18314713 0.18435251 0.16336438]
#      [0.36413198 1.         0.15054075 0.21704584 0.11203887]
#      [0.18314713 0.15054075 1.         0.21318602 0.07763512]
#      [0.18435251 0.21704584 0.21318602 1.         0.12960089]
#      [0.16336438 0.11203887 0.07763512 0.12960089 1.        ]]
----------

# Record start time
start = time.time()

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Print cosine similarity matrix
print(cosine_sim)

# Print time taken
print("Time taken: %s seconds" %(time.time() - start))

-----------
# Record start time
start = time.time()

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Print cosine similarity matrix
print(cosine_sim)

# Print time taken
print("Time taken: %s seconds" %(time.time() - start))
# Notice how both linear_kernel and cosine_similarity produced the same result. However, linear_kernel took a smaller amount of time to execute. When you're working with a very large amount of data and your vectors are in the tf-idf representation, it is good practice to default to linear_kernel to improve performance. (NOTE: In case, you see linear_kernel taking more time, it's because the dataset we're dealing with is extremely small and Python's time module is incapable of capture such minute time differences accurately)
---------

# Initialize the TfidfVectorizer 
tfidf = TfidfVectorizer(stop_words='english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(movie_plots)

# Generate the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
 
# Generate recommendations 
print(get_recommendations('The Dark Knight Rises', cosine_sim, indices))

# You've just built your very first recommendation system. Notice how the recommender correctly identifies 'The Dark Knight Rises' as a Batman movie and recommends other Batman movies as a result. This sytem is, of course, very primitive and there are a host of ways in which it could be improved. One method would be to look at the cast, crew and genre in addition to the plot to generate recommendations. We will not be covering this in this course but you have all the tools necessary to accomplish this. Do give it a try!
-------
# Generate mapping between titles and index
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

def get_recommendations(title, cosine_sim, indices):
    # Get index of movie that matches title
    idx = indices[title]
    # Sort the movies based on the similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores for 10 most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]

# With this recommender function in our toolkit, we are now in a very good place to build the rest of the components of our recommendation engine.
#                title                                            tagline
# 938  Cinema Paradiso  A celebration of youth, friendship, and the ev...
# 630         Spy Hard  All the action. All the women. Half the intell...
# 682        Stonewall                    The fight for the right to love
# 514           Killer                    You only hurt the one you love.
# 365    Jason's Lyric                                   Love is courage.
----------
# Initialize the TfidfVectorizer 
tfidf = TfidfVectorizer(stop_words='english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(transcripts)

# Generate the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
 
# Generate recommendations 
print(get_recommendations('5 ways to kill your dreams', cosine_sim, indices))
# You have successfully built a TED talk recommender. This recommender works surprisingly well despite being trained only on a small subset of TED talks. In fact, three of the talks recommended by our system is also recommended by the official TED website as talks to watch next after '5 ways to kill your dreams'!
----------

# Create the doc object
doc = nlp(sent)

# Compute pairwise similarity scores
for token1 in doc:
  for token2 in doc:
    print(token1.text, token2.text, token1.similarity(token2))
    
# output:
#     I I 1.0
#     I like 0.13463897
#     I apples -0.036133606
#     I and -0.085230574
#     I oranges 0.033708632
#     like I 0.13463897
#     like like 1.0
#     like apples 0.0007651703
#     like and 0.104521796
#     like oranges -0.045859136
#     apples I -0.036133606
#     apples like 0.0007651703
#     apples apples 1.0
#     apples and -0.051072996
#     apples oranges 0.46452007
#     and I -0.085230574
#     and like 0.104521796
#     and apples -0.051072996
#     and and 1.0
#     and oranges 0.038236685
#     oranges I 0.033708632
#     oranges like -0.045859136
#     oranges apples 0.46452007
#     oranges and 0.038236685
#     oranges oranges 1.0
# Notice how the words 'apples' and 'oranges' have the highest pairwaise similarity score. This is expected as they are both fruits and are more related to each other than any other pair of words.
-------------
# Create Doc objects
mother_doc = nlp(mother)
hopes_doc = nlp(hopes)
hey_doc = nlp(hey)

# Print similarity between mother and hopes
print(mother_doc.similarity(hopes_doc))

# Print similarity between mother and hey
print(mother_doc.similarity(hey_doc))

# Notice that 'Mother' and 'Hey You' have a similarity score of 0.9 whereas 'Mother' and 'High Hopes' has a score of only 0.6. This is probably because 'Mother' and 'Hey You' were both songs from the same album 'The Wall' and were penned by Roger Waters. On the other hand, 'High Hopes' was a part of the album 'Division Bell' with lyrics by David Gilmour and his wife, Penny Samson. Treat yourself by listening to these songs. They're some of the best!
---------
