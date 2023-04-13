# Identify the list of words from the choices which do not have the same lemma.
# He, She, I, They
#  All the words listed here are pronouns and will be lemmatized to '-PRON-'.
# Car, Bike, Truck, Bus
# Although all these words refer to vehicles, they are words with distinct base forms.

-----
import spacy

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp("gettysburg")

# Generate the tokens
tokens = [token.text for token in doc]
print(tokens)
# tokens is not correctly defined. In the given code, doc is created with a single word 'gettysburg', which is then used to generate tokens. However, since 'gettysburg' is a single token, the output will be a list containing a single string 'gettysburg', which is not useful for tokenization.
# To correctly tokenize a text, the input doc should contain a string of text with multiple words.
import spacy

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp(gettysburg)

# Generate the tokens
tokens = [token.text for token in doc]
print(tokens)

# ['Four', 'score', 'and', 'seven', 'years', 'ago', 'our', 'fathers', 'brought', 'forth', 'on', 'this', 'continent', ',', 'a', 'new', 'nation', ',', 'conceived', 'in', 'Liberty', ',', 'and', 'dedicated', 'to', 'the', 'proposition', 'that', 'all', 'men', 'are', 'created', 'equal', '.', 'Now', 'we', "'re", 'engaged', 'in', 'a', 'great', 'civil', 'war', ',', 'testing', 'whether', 'that', 'nation', ',', 'or', 'any', 'nation', 'so', 'conceived', 'and', 'so', 'dedicated', ',', 'can', 'long', 'endure', '.', 'We', "'re", 'met', 'on', 'a', 'great', 'battlefield', 'of', 'that', 'war', '.', 'We', "'ve", 'come', 'to', 'dedicate', 'a', 'portion', 'of', 'that', 'field', ',', 'as', 'a', 'final', 'resting', 'place', 'for', 'those', 'who', 'here', 'gave', 'their', 'lives', 'that', 'that', 'nation', 'might', 'live', '.', 'It', "'s", 'altogether', 'fitting', 'and', 'proper', 'that', 'we', 'should', 'do', 'this', '.', 'But', ',', 'in', 'a', 'larger', 'sense', ',', 'we', 'ca', "n't", 'dedicate', '-', 'we', 'can', 'not', 'consecrate', '-', 'we', 'can', 'not', 'hallow', '-', 'this', 'ground', '.', 'The', 'brave', 'men', ',', 'living', 'and', 'dead', ',', 'who', 'struggled', 'here', ',', 'have', 'consecrated', 'it', ',', 'far', 'above', 'our', 'poor', 'power', 'to', 'add', 'or', 'detract', '.', 'The', 'world', 'will', 'little', 'note', ',', 'nor', 'long', 'remember', 'what', 'we', 'say', 'here', ',', 'but', 'it', 'can', 'never', 'forget', 'what', 'they', 'did', 'here', '.', 'It', 'is', 'for', 'us', 'the', 'living', ',', 'rather', ',', 'to', 'be', 'dedicated', 'here', 'to', 'the', 'unfinished', 'work', 'which', 'they', 'who', 'fought', 'here', 'have', 'thus', 'far', 'so', 'nobly', 'advanced', '.', 'It', "'s", 'rather', 'for', 'us', 'to', 'be', 'here', 'dedicated', 'to', 'the', 'great', 'task', 'remaining', 'before', 'us', '-', 'that', 'from', 'these', 'honored', 'dead', 'we', 'take', 'increased', 'devotion', 'to', 'that', 'cause', 'for', 'which', 'they', 'gave', 'the', 'last', 'full', 'measure', 'of', 'devotion', '-', 'that', 'we', 'here', 'highly', 'resolve', 'that', 'these', 'dead', 'shall', 'not', 'have', 'died', 'in', 'vain', '-', 'that', 'this', 'nation', ',', 'under', 'God', ',', 'shall', 'have', 'a', 'new', 'birth', 'of', 'freedom', '-', 'and', 'that', 'government', 'of', 'the', 'people', ',', 'by', 'the', 'people', ',', 'for', 'the', 'people', ',', 'shall', 'not', 'perish', 'from', 'the', 'earth', '.']

# You now know how to tokenize a piece of text. In the next exercise, we will perform similar steps and conduct lemmatization.
-----------
# Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we're engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We're met on a great battlefield of that war. We've come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It's altogether fitting and proper that we should do this. But, in a larger sense, we can't dedicate - we can not consecrate - we can not hallow - this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It's rather for us to be here dedicated to the great task remaining before us - that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion - that we here highly resolve that these dead shall not have died in vain - that this nation, under God, shall have a new birth of freedom - and that government of the people, by the people, for the people, shall not perish from the earth.

# Loop over doc and extract the lemma for each token of gettysburg.
import spacy

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp(gettysburg)

# Generate lemmas
lemmas = [token.lemma_ for token in doc]

--------

import spacy

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp(gettysburg)

# Generate lemmas
lemmas = [token.lemma_ for token in doc]

# Convert lemmas into a string
print(' '.join(lemmas))

# four score and seven year ago our father bring forth on this continent , a new nation , conceive in Liberty , and dedicate to the proposition that all man be create equal . now we be engage in a great civil war , test whether that nation , or any nation so conceive and so dedicated , can long endure . we be meet on a great battlefield of that war . we 've come to dedicate a portion of that field , as a final resting place for those who here give their life that that nation might live . it be altogether fitting and proper that we should do this . but , in a large sense , we ca n't dedicate - we can not consecrate - we can not hallow - this ground . the brave man , live and dead , who struggle here , have consecrate it , far above our poor power to add or detract . the world will little note , nor long remember what we say here , but it can never forget what they do here . it be for we the living , rather , to be dedicate here to the unfinished work which they who fight here have thus far so nobly advanced . it be rather for we to be here dedicate to the great task remain before we - that from these honor dead we take increased devotion to that cause for which they give the last full measure of devotion - that we here highly resolve that these dead shall not have die in vain - that this nation , under God , shall have a new birth of freedom - and that government of the people , by the people , for the people , shall not perish from the earth .

---------

# Load model and create Doc object
nlp = spacy.load('en_core_web_sm')
doc = nlp(blog)

# Generate lemmatized tokens
lemmas = [token.lemma_ for token in doc]

# Remove stopwords and non-alphabetic tokens
a_lemmas = [lemma for lemma in lemmas 
            if lemma.isalpha() and lemma not in stopwords]

# Print string after text cleaning
print(' '.join(a_lemmas))
# Take a look at the cleaned text; it is lowercased and devoid of numbers, punctuations and commonly used stopwords. Also, note that the word U.S. was present in the original text. Since it had periods in between, our text cleaning process completely removed it. This may not be ideal behavior. It is always advisable to use your custom functions in place of isalpha() for more nuanced cases.
-------

# Function to preprocess text
def preprocess(text):
  	# Create Doc object
    doc = nlp(text, disable=['ner', 'parser'])
    # Generate lemmas
    lemmas = [token.lemma_ for token in doc]
    # Remove stopwords and non-alphabetic characters
    a_lemmas = [lemma for lemma in lemmas 
            if lemma.isalpha() and lemma not in stopwords]
    
    return ' '.join(a_lemmas)
  
# Apply preprocess to ted['transcript']
ted['transcript'] = ted['transcript'].apply(preprocess)
print(ted['transcript'])
# You have preprocessed all the TED talk transcripts contained in ted and it is now in a good shape to perform operations such as vectorization (as we will soon see how). You now have a good understanding of how text preprocessing works and why it is important. In the next lessons, we will move on to generating word level features for our texts.
----------

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp(lotf)

# Generate tokens and pos tags
pos = [(token.text, token.pos_) for token in doc]
print(pos)
# output:
#     [('He', 'PRON'), ('found', 'VERB'), ('himself', 'PRON'), ('understanding', 'VERB'), ('the', 'DET'), ('wearisomeness', 'NOUN'), ('of', 'ADP'), ('this', 'DET'), ('life', 'NOUN'), (',', 'PUNCT'), ('where', 'ADV'), ('every', 'DET'), ('path', 'NOUN'), ('was', 'VERB'), ('an', 'DET'), ('improvisation', 'NOUN'), ('and', 'CCONJ'), ('a', 'DET'), ('considerable', 'ADJ'), ('part', 'NOUN'), ('of', 'ADP'), ('one', 'PRON'), ('’s', 'ADV'), ('waking', 'VERB'), ('life', 'NOUN'), ('was', 'AUX'), ('spent', 'VERB'), ('watching', 'VERB'), ('one', 'NUM'), ('’s', 'NOUN'), ('feet', 'NOUN'), ('.', 'PUNCT')]

# Examine the various POS tags attached to each token and evaluate if they make intuitive sense to you. You will notice that they are indeed labelled correctly according to the standard rules of English grammar.
-------
nlp = spacy.load('en_core_web_sm')

# Returns number of proper nouns
def proper_nouns(text, model=nlp):
  	# Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]
    
    # Return number of proper nouns
    return pos.count('PROPN')

print(proper_nouns("Abdul, Bill and Cathy went to the market to buy apples.", nlp))
---------
nlp = spacy.load('en_core_web_sm')

# Returns number of other nouns
def nouns(text, model=nlp):
  	# Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]
    
    # Return number of other nouns
    return pos.count('NOUN')

print(nouns("Abdul, Bill and Cathy went to the market to buy apples.", nlp))

# You now know how to write functions that compute the number of instances of a particulat POS tag in a given piece of text. In the next exercise, we will use these functions to generate features from text in a dataframe.
-------

headlines['num_propn'] = headlines['title'].apply(proper_nouns)

# Compute mean of proper nouns
real_propn = headlines[headlines['label'] == 'REAL']['num_propn'].mean()
fake_propn = headlines[headlines['label'] == 'FAKE']['num_propn'].mean()

# Print results
print("Mean no. of proper nouns in real and fake headlines are %.2f and %.2f respectively"%(real_propn, fake_propn))

---------
headlines['num_noun'] = headlines['title'].apply(nouns)

# Compute mean of other nouns
real_noun = headlines[headlines['label'] == 'REAL']['num_noun'].mean()
fake_noun = headlines[headlines['label'] == 'FAKE']['num_noun'].mean()

Print results
print("Mean no. of other nouns in real and fake headlines are %.2f and %.2f respectively"%(real_noun, fake_noun))

# You now know to construct features using POS tags information. Notice how the mean number of proper nouns is considerably higher for fake news than it is for real news. The opposite seems to be true in the case of other nouns. This fact can be put to great use in designing fake news detectors.
----------
# Load the required model
nlp = spacy.load('en_core_web_sm')

# Create a Doc instance 
text = 'Sundar Pichai is the CEO of Google. Its headquarters is in Mountain View.'
doc = nlp(text)

# Print all named entities and their labels
for ent in doc.ents:
    print(ent.text, ent.label_)

# output:
#     Google ORG
#     Mountain View GPE
# Notice how the model correctly predicted the labels of Google and Mountain View but mislabeled Sundar Pichai as an organization. As discussed in the video, the predictions of the model depend strongly on the data it is trained on. It is possible to train spaCy models on your custom data. You will learn to do this in more advanced NLP courses.
--------

def find_persons(text):
  # Create Doc object
  doc = nlp(text)
  
  # Identify the persons
  persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
  
  # Return persons
  return persons

print(find_persons(tc))
#  output:
#     ['Sheryl Sandberg', 'Mark Zuckerberg']
# The article was related to Facebook and our function correctly identified both the people mentioned. You can now see how NER could be used in a variety of applications. Publishers may use a technique like this to classify news articles by the people mentioned in them. A question answering system could also use something like this to answer questions such as 'Who are the people mentioned in this passage?'. With this, we come to an end of this chapter. In the next, we will learn how to conduct vectorization on documents.
------

---------
