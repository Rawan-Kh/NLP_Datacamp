# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Print the names of the pipeline components
print(nlp.pipe_names)

# Print the full pipeline of (name, component) tuples
print(nlp.pipeline)

# Whenever you're unsure about the current pipeline, you can inspect it by printing nlp.pipe_names or nlp.pipeline
# output:
#     ['tagger', 'parser', 'ner']
#     [('tagger', <spacy.pipeline.pipes.Tagger object at 0x7f19a7ed5b00>), ('parser', <spacy.pipeline.pipes.DependencyParser object at 0x7f198cf27ac8>), ('ner', <spacy.pipeline.pipes.EntityRecognizer object at 0x7f198cf27b28>)]
------------

# Custom components are great for adding custom values to documents, tokens and spans, and customizing the doc.ents.
---------
# Define the custom component
def length_component(doc):
    # Get the doc's length
    doc_length = len(doc)
    print("This document is {} tokens long.".format(doc_length))
    # Return the doc
    return doc
  
# Load the small English model and Add the component first in the pipeline
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe(length_component, first=True)

# Process a text
doc = nlp("Hello there")
# Now let's take a look at a slightly more complex component!
-----------

# Define the custom component
def animal_component(doc):
    # Create a Span for each match and assign the label 'ANIMAL'
    # and overwrite the doc.ents with the matched spans
    doc.ents = [Span(doc, start, end, label='ANIMAL')
                for match_id, start, end in matcher(doc)]
    return doc
    
# Add the component to the pipeline after the 'ner' component 
nlp.add_pipe(animal_component, after='ner')

# Process the text and print the text and label for the doc.ents
doc = nlp("I have a cat and a Golden Retriever")
print([(ent.text, ent.label_) for ent in doc.ents])

# output:
#     ['tagger', 'parser', 'ner', 'animal_component']
# animal_patterns: [Golden Retriever, cat, turtle, Rattus norvegicus]

# <script.py> output:
#     [('cat', 'ANIMAL'), ('Golden Retriever', 'ANIMAL')]
# You've built your first pipeline component for rule-based entity matching
-----------

# Register the Token extension attribute 'is_country' with the default value False
Token.set_extension('is_country', default=False)

# Process the text and set the is_country attribute to True for the token "Spain"
doc = nlp("I live in Spain.")
doc[3]._.is_country = True

# Print the token text and the is_country attribute for all tokens
print([( token.text, token._.is_country) for token in doc])
# [('I', False), ('live', False), ('in', False), ('Spain', True), ('.', False)]
-----------
# Define the getter function that takes a token and returns its reversed text
def get_reversed(token):
    return token.text[::-1]
  
# Register the Token property extension 'reversed' with the getter get_reversed
Token.set_extension('reversed', getter=get_reversed )

# Process the text and print the reversed attribute for each token
doc = nlp("All generalizations are false, including this one.")
for token in doc:
    print('reversed:', token._.reversed)

# <script.py> output:
#     reversed: llA
#     reversed: snoitazilareneg
#     reversed: era
#     reversed: eslaf
#     reversed: ,
#     reversed: gnidulcni
#     reversed: siht
#     reversed: eno
#     reversed: .
------------------
# Define the getter function
def get_has_number(doc):
    # Return if any of the tokens in the doc return True for token.like_num
    return any(token.like_num for token in doc)

# Register the Doc property extension 'has_number' with the getter get_has_number
Doc.set_extension('has_number', getter=get_has_number)

# Process the text and check the custom has_number attribute 
doc = nlp("The museum closed for five years in 2012.")
print('has_number:', doc._.has_number)
# output:
#     has_number: True
-----------

# Define the method
def to_html(span, tag):
    # Wrap the span text in a HTML tag and return it
    return '<{tag}>{text}</{tag}>'.format(tag=tag, text=span.text)

# Process the text and call the to_html method on the span with the tag name 'strong'
doc = nlp("Hello world, this is a sentence.")
span = doc[0:2]

# Register the Span property extension 'to_html' with the method to_html
span.set_extension('to_html', method=to_html)

print(span._.to_html('strong'))
# output:
#     <strong>Hello world</strong>
# In the next exercise, you'll get to combine custom attributes with custom pipeline components.
----------

def get_wikipedia_url(span):
    # Get a Wikipedia URL if the span has one of the labels
    if span.label_ in ('PERSON', 'ORG', 'GPE', 'LOCATION'):
        entity_text = span.text.replace(' ', '_')
        return "https://en.wikipedia.org/w/index.php?search=" + entity_text

# Set the Span extension wikipedia_url using get getter get_wikipedia_url
Span.set_extension('wikipedia_url', getter=get_wikipedia_url)

doc = nlp("In over fifty years from his very first recordings right through to his last album, David Bowie was at the vanguard of contemporary culture.")
for ent in doc.ents:
    # Print the text and Wikipedia URL of the entity
    print( ent.text, ent._.wikipedia_url)
 
# output:
#     over fifty years None
#     first None
#     David Bowie https://en.wikipedia.org/w/index.php?search=David_Bowie
# ou now have a pipeline component that uses named entities predicted by the model to generate Wikipedia URLs and adds them as a custom attribute. Try opening the link in your browser to see what happens!
-----------
def countries_component(doc):
    # Create an entity Span with the label 'GPE' for all matches
    doc.ents = [Span(doc, start, end, label='GPE')
                for match_id, start, end in matcher(doc)]
    return doc

# Add the component to the pipeline
nlp.add_pipe(countries_component)

# Register capital and getter that looks up the span text in country capitals
Span.set_extension('capital', getter=lambda span: capitals.get(span.text))

# Process the text and print the entity text, label and capital attributes
doc = nlp("Czech Republic may help Slovakia protect its airspace")
print([(ent.text, ent.label_, ent._.capital) for ent in doc.ents])
# output:
#     [('Czech Republic', 'GPE', 'Prague'), ('Slovakia', 'GPE', 'Bratislava')]
# This is a great example of how you can add structured data to your spaCy pipeline.
----------
# # Process the texts and print the adjectives
# for text in TEXTS:
#     doc = nlp(text)
#     print([token.text for token in doc if token.pos_ == 'ADJ'])

# Process the texts and print the adjectives
for doc in nlp.pipe(TEXTS):
    print([token.text for token in doc if token.pos_ == "ADJ"])
# output:
#     ['favorite']
#     ['sick']
#     []
#     ['happy']
#     ['delicious', 'fast']
#     []
#     ['terrible', 'gettin', 'payin']
--------

# Process the texts and print the entities
# docs = [nlp(text) for text in TEXTS]
# entities = [doc.ents for doc in docs]
# print(*entities)

docs = list(nlp.pipe(TEXTS))
entities = [doc.ents for doc in docs]
print(*entities)

#  output:
#     (McDonalds,) (@McDonalds,) (McDonalds,) (McDonalds, Spain) (The Arch Deluxe,) (WANT, McRib) (This morning,)

-------------

people = ['David Bowie', 'Angela Merkel', 'Lady Gaga']

# Create a list of patterns for the PhraseMatcher
# patterns = [nlp(person) for person in people]

patterns = list(nlp.pipe(people))
patterns

[David Bowie, Angela Merkel, Lady Gaga]

# Let's move on to a practical example that uses nlp.pipe to process documents with additional meta data.
--------
# Import the Doc class and register the extensions 'author' and 'book'
from spacy.tokens import Doc
Doc.set_extension('book', default=None)
Doc.set_extension('author', default=None)

for doc, context in nlp.pipe(DATA, as_tuples=True):
    # Set the doc._.book and doc._.author attributes from the context
    doc._.book = context['book']
    doc._.author = context['author']
    
    # Print the text and custom attribute data
    print(doc.text, '\n', "— '{}' by {}".format(doc._.book, doc._.author), '\n')
    
# output:
#     One morning, when Gregor Samsa woke from troubled dreams, he found himself transformed in his bed into a horrible vermin. 
#      — 'Metamorphosis' by Franz Kafka 
    
#     I know not all that may be coming, but be it what it will, I'll go to it laughing. 
#      — 'Moby-Dick or, The Whale' by Herman Melville 
    
#     It was the best of times, it was the worst of times. 
#      — 'A Tale of Two Cities' by Charles Dickens 
    
#     The only people for me are the mad ones, the ones who are mad to live, mad to talk, mad to be saved, desirous of everything at the same time, the ones who never yawn or say a commonplace thing, but burn, burn, burn like fabulous yellow roman candles exploding like spiders across the stars. 
#      — 'On the Road' by Jack Kerouac 
    
#     It was a bright cold day in April, and the clocks were striking thirteen. 
#      — '1984' by George Orwell 
    
#     Nowadays people know the price of everything and the value of nothing. 
#      — 'The Picture Of Dorian Gray' by Oscar Wilde  

# The same technique is useful for a variety of tasks. For example, you could pass in page or paragraph numbers to relate the processed Doc back to the position in a larger document. Or you could pass in other structured data like IDs referring to a knowledge base.
----------

text = "Chick-fil-A is an American fast food restaurant chain headquartered in the city of College Park, Georgia, specializing in chicken sandwiches."

# Disable the tagger and parser
with nlp.disable_pipes('tagger', 'parser'):
    # Process the text
    doc = nlp(text)
    # Print the entities in the doc
    print(doc.ents)
 
# output:
#     (American, College Park, Georgia)
------------
# Now that you've practiced the performance tips and tricks, you're ready for the next chapter and training spaCy's neural network models.
