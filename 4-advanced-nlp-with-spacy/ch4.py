# That's right. spaCy's components are supervised models for text annotations, meaning they can only learn to reproduce examples, not guess new labels from raw text
---------
# Two tokens whose lowercase forms match 'iphone' and 'x'
pattern1 = [{"LOWER": "iphone"}, {"LOWER": "x"}]

# Token whose lowercase form matches 'iphone' and an optional digit
pattern2 = [{"LOWER": "iphone"}, {"IS_DIGIT": True, "OP": "?"}]

# Add patterns to the matcher
matcher.add('GADGET', None, pattern1, pattern2)

# Now let's use those patterns to quickly bootstrap some training data for our model.
------------
# Create a Doc object for each text in TEXTS
for text in TEXTS:
    doc = nlp(text)
    # Find the matches in the doc
    matches = matcher(doc)
    
    # Get a list of (start, end, label) tuples of matches in the text
    entities = [(start, end, 'GADGET') for match_id, start, end in matches]
    print(doc.text, entities)
# How to preorder the iPhone X [(4, 6, 'GADGET')]
# iPhone X is coming [(0, 2, 'GADGET')]
# Should I pay $1,000 for the iPhone X? [(7, 9, 'GADGET')]
# The iPhone 8 reviews are here [(1, 3, 'GADGET')]
# Your iPhone goes up to 11 today []
# I need a new phone! Any tips? []
----------

# Create a blank 'en' model
nlp = spacy.blank('en')

# Create a new entity recognizer and add it to the pipeline
ner = nlp.create_pipe('ner')
nlp.add_pipe(ner)

# Add the label 'GADGET' to the entity recognizer
ner.add_label('GADGET')
# The pipeline is now ready, so let's start writing the training loop.
---------
# Start the training
nlp.begin_training()

# Loop for 10 iterations
for itn in range(10):
    # Shuffle the training data
    random.shuffle(TRAINING_DATA)
    losses = {}
    
    # Batch the examples and iterate over them
    for batch in spacy.util.minibatch(TRAINING_DATA, size=2):
        texts = [text for text, entities in batch]
        annotations = [entities for text, entities in batch]
        
        # Update the model
        nlp.update(texts, annotations, losses=losses)
        print(losses)
   
#  you've successfully trained your first spaCy model. The numbers printed to the IPython shell represent the loss on each iteration, the amount of work left for the optimizer. The lower the number, the better. In real life, you normally want to use a lot more data than this, ideally at least a few hundred or a few thousand examples.
# output:
#     {'ner': 12.799999475479126}
#     {'ner': 24.160905599594116}
#     {'ner': 31.979278087615967}
#     {'ner': 6.113706707954407}
#     {'ner': 12.014856994152069}
#     {'ner': 17.614861756563187}
#     {'ner': 1.8505476117134094}
#     {'ner': 5.22869636118412}
#     {'ner': 10.294682228937745}
#     {'ner': 1.2793622845201753}
#     {'ner': 5.1642187417601235}
#     {'ner': 7.097947950736852}
#     {'ner': 4.471738298423588}
#     {'ner': 6.75384229251722}
#     {'ner': 9.632959776721691}
#     {'ner': 3.400470433756709}
#     {'ner': 4.594839826455427}
#     {'ner': 7.1772303527650365}
#     {'ner': 1.7282676775939763}
#     {'ner': 3.469022838282399}
#     {'ner': 3.709058039627962}
#     {'ner': 0.10645238201323082}
#     {'ner': 0.22150556282213074}
#     {'ner': 2.7843695867391034}
#     {'ner': 0.021411064453332074}
#     {'ner': 0.025004193937547825}
#     {'ner': 1.3728112514579607}
#     {'ner': 1.0639168628903235}
#     {'ner': 1.0650975566521197}
#     {'ner': 1.0671759424469656}

----------

# Process each text in TEST_DATA
for doc in nlp.pipe(TEST_DATA):
    # Print the document text and entitites
    print(doc.text)
    print(doc.ents, '\n\n')
 
# output:
#     Apple is slowing down the iPhone 8 and iPhone X - how to stop it
#     (iPhone 8, iPhone X) 
    
    
#     I finally understand what the iPhone X 'notch' is for
#     (iPhone X,) 
    
    
#     Everything you need to know about the Samsung Galaxy S9
#     (Samsung Galaxy,) 
    
    
#     Looking to compare iPad models? Here’s how the 2018 lineup stacks up
#     (iPad,) 
    
    
#     The iPhone 8 and iPhone 8 Plus are smartphones designed, developed, and marketed by Apple
#     (iPhone 8, iPhone 8) 
    
    
#     what is the cheapest ipad, especially ipad pro???
#     (ipad, ipad) 
    
    
#     Samsung Galaxy is a series of mobile computing devices designed, manufactured and marketed by Samsung Electronics
#     (Samsung Galaxy,) 
# On our test data, the model achieved an accuracy of 70%.
-----------
TRAINING_DATA = []

# Create a Doc object for each text in TEXTS
for doc in nlp.pipe(TEXTS):
    # Match on the doc and create a list of matched spans
    spans = [doc[start:end] for match_id, start, end in matcher(doc)]
    # Get (start character, end character, label) tuples of matches
    entities = [(span.start_char, span.end_char, 'GADGET') for span in spans]
    
    # Format the matches as a (doc.text, entities) tuple
    training_example = (doc.text, {'entities': entities})
    # Append the example to the training data
    TRAINING_DATA.append(training_example)
    
print(*TRAINING_DATA, sep='\n') 
# output:
#     ('How to preorder the iPhone X', {'entities': [(20, 28, 'GADGET')]})
#     ('iPhone X is coming', {'entities': [(0, 8, 'GADGET')]})
#     ('Should I pay $1,000 for the iPhone X?', {'entities': [(28, 36, 'GADGET')]})
#     ('The iPhone 8 reviews are here', {'entities': [(4, 12, 'GADGET')]})
#     ('Your iPhone goes up to 11 today', {'entities': []})
#     ('I need a new phone! Any tips?', {'entities': []})

# Before you train a model with the data, you always want to double-check that your matcher didn't identify any false positives. But that process is still much faster than doing everything manually.
------------
TRAINING_DATA = [
    ("i went to amsterdem last year and the canals were beautiful", {'entities': [(10, 19, 'GPE')]}),
    ("You should visit Paris once in your life, but the Eiffel Tower is kinda boring", {'entities': [(17, 22, 'GPE')]}),
    # find out where the entity span starts and where it ends. Then add (start, end, label) tuples to the entities.
    ("There's also a Paris in Arkansas, lol", {'entities': [(15, 20, "GPE"), (24, 32, "GPE")]}),
    ("Berlin is perfect for summer holiday: lots of parks, great nightlife, cheap beer!", {'entities': [(0, 6, 'GPE')]})
]
     
print(*TRAINING_DATA, sep='\n')
#  output:
#     ('i went to amsterdem last year and the canals were beautiful', {'entities': [(10, 19, 'GPE')]})
#     ('You should visit Paris once in your life, but the Eiffel Tower is kinda boring', {'entities': [(17, 22, 'GPE')]})
#     ("There's also a Paris in Arkansas, lol", {'entities': [(15, 20, 'GPE'), (24, 32, 'GPE')]})
#     ('Berlin is perfect for summer holiday: lots of parks, great nightlife, cheap beer!', {'entities': [(0, 6, 'GPE')]})
# Once the model achieves good results on detecting GPE entities in the traveler reviews, you could add a rule-based component to determine whether the entity is a tourist destination in this context. For example, you could resolve the entities types back to a knowledge base or look them up in a travel wiki.
--------------

TRAINING_DATA = [
    ("Reddit partners with Patreon to help creators build communities", 
     {'entities': [(0, 6, "WEBSITE"), (21, 28, "WEBSITE")]}),
  
    ("PewDiePie smashes YouTube record", 
     {'entities': [(18, 25, "WEBSITE")]}),
  
    ("Reddit founder Alexis Ohanian gave away two Metallica tickets to fans", 
     {'entities': [(0, 6, "WEBSITE")]}),
    # And so on...
]
----------
TRAINING_DATA = [
    ("Reddit partners with Patreon to help creators build communities", 
     {'entities': [(0, 6, 'WEBSITE'), (21, 28, 'WEBSITE')]}),
  
    ("PewDiePie smashes YouTube record", 
     {'entities': [(0, 9, "PERSON"), (18, 25, 'WEBSITE')]}),
  
    ("Reddit founder Alexis Ohanian gave away two Metallica tickets to fans", 
     {'entities': [(0, 6, 'WEBSITE'), (15, 29, "PERSON")]}),
    # And so on...
]

# After including both examples of the next WEBSITE entities, as well as existing entity types like PERSON, the model now performs much better.
