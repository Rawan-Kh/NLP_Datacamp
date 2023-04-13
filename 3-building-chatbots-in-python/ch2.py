# Define a dictionary of patterns
patterns = {}

# Iterate over the keywords dictionary
for intent , keys in keywords.items():
    # Create regular expressions and compile them into pattern objects
    patterns[intent] = re.compile('|'.join(keys))
    
# Print the patterns
print(patterns)

# The next step is to define a function to find the intent of a message.
----------

# Define a function to find the intent of a message
def match_intent(message):
    matched_intent = None
    for intent, pattern in patterns.items():
        # Check if the pattern occurs in the message 
        if pattern.search(message):
            matched_intent =intent
    return matched_intent

# Define a respond function
def respond(message):
    # Call the match_intent function
    intent = match_intent(message)
    # Fall back to the default response
    key = "default"
    if intent in responses:
        key = intent
    return responses[key]

# Send messages
send_message("hello!")
send_message("bye byeee")
send_message("thanks very much!")
# Your bot can now respond to messages even when they don't exactly match.
------------

# Define find_name()
def find_name(message):
    name = None
    # Create a pattern for checking if the keywords occur
    name_keyword = re.compile('name|call')
    # Create a pattern for finding capitalized words
    name_pattern = re.compile("[A-Z]{1}[a-z]*")
    if name_keyword.search(message):
        # Get the matching words in the string
        name_words = name_pattern.findall(message)
        if len(name_words) > 0:
            # Return the name if the keywords are present
            name = ' '.join(name_words)
    return name

# Define respond()
def respond(message):
    # Find the name
    name = find_name(message)
    if name is None:
        return "Hi there!"
    else:
        return "Hello, {0}!".format(name)

# Send messages
send_message("my name is David Copperfield")
send_message("call me Ishmael")
send_message("People call me Cassandra")

# You just built a simple entity recognizer using regex. However, as you can see with the final output of send_message(), the mix of using regex while making assumptions does have its limitations
# output:
#     USER : my name is David Copperfield
#     BOT : Hello, David Copperfield!
#     USER : call me Ishmael
#     BOT : Hello, Ishmael!
#     USER : People call me Cassandra
#     BOT : Hello, People Cassandra!
-------------

# Load the spacy model: nlp
nlp = spacy.load('en')

# Calculate the length of sentences
n_sentences = len(sentences)

# Calculate the dimensionality of nlp
embedding_dim = nlp.vocab.vectors_length

# Initialize the array with zeros: X
X = np.zeros((n_sentences, embedding_dim))

# Iterate over the sentences
for idx, sentence in enumerate(sentences):
    # Pass each each sentence to the nlp object to create a document
    doc = nlp(sentence)
    # Save the document's .vector attribute to the corresponding row in X
    X[idx, :] = doc.vector

# You can now find vector representations of words and sentences with spaCy.    
----------

# Import SVC
from sklearn.svm import SVC

# Create a support vector classifier
clf = SVC(C=1)

# Fit the classifier using the training data
clf.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = clf.predict(X_test)

# Count the number of correct predictions
n_correct = 0
for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        n_correct += 1

print("Predicted {0} correctly out of {1} test examples".format(n_correct, len(y_test)))
# You just trained a support vector machine for recognizing intents
------------
# Define included_entities
include_entities = ['DATE', 'ORG', 'PERSON']

# Define extract_entities()
def extract_entities(message):
    # Create a dict to hold the entities
    ents = dict.fromkeys(include_entities)
    # Create a spacy document
    doc = nlp(message)
    for ent in doc.ents:
        if ent.label_ in include_entities:
            # Save interesting entities
            ents[ent.label_] = ent.text
    return ents

print(extract_entities('friends called Mary who have worked at Google since 2010'))
print(extract_entities('people who graduated from MIT in 1999'))

# Great work. Your bot can now make use of spaCy's built-in entity recognizer.
# output:
#     {'DATE': '2010', 'ORG': None, 'PERSON': None}
#     {'DATE': None, 'ORG': None, 'PERSON': None}
-----------------

# Create the document
doc = nlp("let's see that jacket in red and some blue jeans")

# Iterate over parents in parse tree until an item entity is found
def find_parent_item(word):
    # Iterate over the word's ancestors
    for parent in word.ancestors:
        # Check for an "item" entity
        if entity_type(parent) == "item":
            return parent.text
    return None

# For all color entities, find their parent item
def assign_colors(doc):
    # Iterate over the document
    for word in doc:
        # Check for "color" entities
        if entity_type(word) == "color":
            # Find the parent
            item =  find_parent_item(word)
            print("item: {0} has color : {1}".format(item, word))

# Assign the colors
assign_colors(doc) 

# output:
#     item: jacket has color : red
#     item: jeans has color : blue
# Your bot can now figure out simple relationships between entities.
-----------

# Import necessary modules
from rasa_nlu.converters import load_data
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer

# Create args dictionary
args = {"pipeline": "spacy_sklearn"}

# Create a configuration and trainer
config = RasaNLUConfig(cmdline_args=args)
trainer = Trainer(config)

# Load the training data
training_data = load_data("./training_data.json")

# Create an interpreter by training the model
interpreter = trainer.train(training_data)

# Test the interpreter
print(interpreter.parse("I'm looking for a Mexican restaurant in the North of town"))

# Congrats! You just trained an intent and entity recognizer without having to create any arrays.
# output:
#     Fitting 2 folds for each of 6 candidates, totalling 12 fits
# {'intent': {'name': 'restaurant_search', 'confidence': 0.6627604390878398}, 'entities': [{'start': 18, 'end': 25, 'value': 'mexican', 'entity': 'cuisine', 'extractor': 'ner_crf'}, {'start': 44, 'end': 49, 'value': 'north', 'entity': 'location', 'extractor': 'ner_crf'}], 'intent_ranking': [{'name': 'restaurant_search', 'confidence': 0.6627604390878398}, {'name': 'goodbye', 'confidence': 0.14633725788681204}, {'name': 'affirm', 'confidence': 0.09756426473688806}, {'name': 'greet', 'confidence': 0.09333803828846025}], 'text': "I'm looking for a Mexican restaurant in the North of town"}
-----------

# Import necessary modules
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer

pipeline = [
    "nlp_spacy",
    "tokenizer_spacy",
    "ner_crf"
]

# Create a config that uses this pipeline
config = RasaNLUConfig(cmdline_args={"pipeline": pipeline})

# Create a trainer that uses this config
trainer = Trainer(config)

# Create an interpreter by training the model
interpreter = trainer.train(training_data)

# Parse some messages
print(interpreter.parse("show me Chinese food in the centre of town"))
print(interpreter.parse("I want an Indian restaurant in the west"))
print(interpreter.parse("are there any good pizza places in the center?"))

# You just built a custom entity recogniser with Rasa NLU.
# output:
#     {'intent': {'name': '', 'confidence': 0.0}, 'entities': [{'start': 28, 'end': 34, 'value': 'centre', 'entity': 'location', 'extractor': 'ner_crf'}], 'text': 'show me Chinese food in the centre of town'}
#     {'intent': {'name': '', 'confidence': 0.0}, 'entities': [{'start': 10, 'end': 16, 'value': 'indian', 'entity': 'cuisine', 'extractor': 'ner_crf'}, {'start': 35, 'end': 39, 'value': 'west', 'entity': 'location', 'extractor': 'ner_crf'}], 'text': 'I want an Indian restaurant in the west'}
#     {'intent': {'name': '', 'confidence': 0.0}, 'entities': [{'start': 39, 'end': 45, 'value': 'center', 'entity': 'location', 'extractor': 'ner_crf'}], 'text': 'are there any good pizza places in the center?'}
-----------
