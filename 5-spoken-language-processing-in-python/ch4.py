# Create function to convert audio file to wav
def convert_to_wav(filename):
  """Takes an audio file of non .wav format and converts to .wav"""
  # Import audio file
  audio = AudioSegment.from_file(filename)
  
  # Create new filename
  new_filename = filename.split(".")[0] + ".wav"
  
  # Export file as .wav
  audio.export(new_filename, format='wav')
  print(f"Converting {filename} to {new_filename}...")
 
# Test the function
convert_to_wav("call_1.mp3")  #takes "call_1.mp3" not 'call_1.mp3'
# The first function down! Beautiful. Now to convert any audio file to .wav format, you can pass the filename to convert_to_wav(). Creating functions like this at the start of your projects saves plenty of coding later on.
---------

def show_pydub_stats(filename):
  """Returns different audio attributes related to an audio file."""
  # Create AudioSegment instance
  audio_segment = AudioSegment.from_file(filename)
  
  # Print audio attributes and return AudioSegment instance
  print(f"Channels: {audio_segment.channels}")
  print(f"Sample width: {audio_segment.sample_width}")
  print(f"Frame rate (sample rate): {audio_segment.frame_rate}")
  print(f"Frame width: {audio_segment.frame_width}")
  print(f"Length (ms): {len(audio_segment)}")
  return audio_segment

# Try the function
call_1_audio_segment = show_pydub_stats("call_1.wav")
# output:
#     Channels: 2
#     Sample width: 2
#     Frame rate (sample rate): 32000
#     Frame width: 4
#     Length (ms): 54888
# Now you'll be able to find the PyDub attribute parameters of any audio file in one line! It seems call_1.wav has two channels, potentially they could be split using PyDubs's split_to_mono() and transcribed separately.
------------

def transcribe_audio(filename):
  """Takes a .wav format audio file and transcribes it to text."""
  # Setup a recognizer instance
  recognizer = sr.Recognizer()
  
  # Import the audio file and convert to audio data
  audio_file = sr.AudioFile(filename)
  with audio_file as source:
    audio_data = recognizer.record(source)
  
  # Return the transcribed text
  return recognizer.recognize_google(audio_data)

# Test the function
print(transcribe_audio("call_1.wav"))
# output:
#     hello welcome to Acme studio support line my name is Daniel how can I best help you hey Daniel this is John I've recently bought a smart from you guys 3 weeks ago and I'm already having issues with it I know that's not good to hear John let's let's get your cell number and then we can we can set up a way to fix it for you one number for 17 varies how long do you reckon this is going to try our best to get the steel number will start up this support case I'm just really really really really I've been trying to contact past three 4 days now and I've been put on hold more than an hour and a half so I'm not really happy I kind of wanna get this issue 6 is f***** possible

# You'll notice the recognizer didn't transcribe the words 'fast as' adequately on the last line, starring them out as a potential expletive, this is a reminder speech recognition still isn't perfect. But now you've now got a function which can transcribe the audio of a .wav file with one line of code. They're a bit of effort to setup but once you've got them, helper functions like transcribe_audio() save time and prevent errors later on.
----------
# Convert mp3 file to wav
convert_to_wav("call_1.mp3")

# Check the stats of new file
call_1 = show_pydub_stats("call_1.wav")

# Split call_1 to mono
call_1_split = call_1.split_to_mono()

# Export channel 2 (the customer channel)
call_1_split[1].export("call_1_channel_2.wav",
                       format="wav")

# Transcribe the single channel
print(transcribe_audio(call_1_split[1]))
# output:
#     Converting call_1.mp3 to call_1.wav...
#     Channels: 2
#     Sample width: 2
#     Frame rate (sample rate): 32000
#     Frame width: 4
#     Length (ms): 54888
#     hey Daniel this is John I've recently bought a smartphone from you guys 3 weeks ago and I'm already having issues with it once they can we grab my Siri number it is for 1757 and very displease how long do you reckon this is going to take a pee on hold for about an hour now right I'm just just really really really really just weasel this product I've been trying to contact supports the past past three 4 days now and have been put on hold for more than an hour and a half so I'm not really happy I kind of wanna get this issue fixed as fast as possible
#  Now we've got some ways to turn our audio files into text, let's use some natural language processing to find out more.
---------
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create SentimentIntensityAnalyzer instance
sid = SentimentIntensityAnalyzer()

# Let's try it on one of our phone calls
call_2_text = transcribe_audio('call_2.wav')

# Display text and sentiment polarity scores
print(call_2_text)
print(sid.polarity_scores(call_2_text))
# output:
#     hello my name is Daniel thank you for calling acne Studios how can I best help you a little bit more but I'm corner of Edward and Elizabeth according to Google according to the match but would you be able to help me in some way because I think I'm actually walk straight past your shop yeah sure thing or thank you so it's good to hear you're enjoying it let me find out where the nearest store is for you
#     {'neg': 0.0, 'neu': 0.694, 'pos': 0.306, 'compound': 0.9817}
# Consider it analyzed! Reading back the transcribed text and listening to the phone call, a compound score of close to 1 (more positive) makes sense since the customer states they're very happy and enjoying their device. Let's keep going!
----------

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create SentimentIntensityAnalyzer instance
sid = SentimentIntensityAnalyzer()

# Transcribe customer channel of call 2
call_2_channel_2_text = transcribe_audio('call_2.wav')

# Display text and sentiment polarity scores
print(call_2_channel_2_text)
print(sid.polarity_scores(call_2_channel_2_text))

-----------
# Import sent_tokenize from nltk
from nltk import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create SentimentIntensityAnalyzer instance
sid = SentimentIntensityAnalyzer()

# Split call 2 channel 2 into sentences and score each
for sentence in sent_tokenize(call_2_channel_2_text):
    print(sentence)
    print(sid.polarity_scores(sentence))
# output:
#     oh hi Daniel my name is Sally I recently purchased a smartphone from you guys and extremely happy with it I've just gotta issue not an issue but I've just got to learn a little bit more about the message bank on I have Google the location but I'm I'm finding it hard I thought you were on the corner of Edward and Elizabeth according to Google according to the match but would you be able to help me in some way because I think I've actually walk straight past your shop
#     {'neg': 0.017, 'neu': 0.891, 'pos': 0.091, 'compound': 0.778}    
-----------
# Import sent_tokenize from nltk
from nltk import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create SentimentIntensityAnalyzer instance
sid = SentimentIntensityAnalyzer()

# Split channel 2 paid text into sentences and score each
for sentence in sent_tokenize(call_2_channel_2_paid_api_text):
    print(sentence)
    print(sid.polarity_scores(sentence))
#  That's pretty cool, you can see how the sentiment differs from sentence to sentence in the call 2 channel 2 paid API text. An extension could be to dig deeper into each of the sentences which have the lowest scores. Let's push on!
# output:
#     Hello and welcome to acme studios.
#     {'neg': 0.0, 'neu': 0.625, 'pos': 0.375, 'compound': 0.4588}
#     My name's Daniel.
#     {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
#     How can I best help you?
#     {'neg': 0.0, 'neu': 0.303, 'pos': 0.697, 'compound': 0.7845}
#     Hi Diane.
#     {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
#     This is paid on this call up to see the status of my, I'm proctor mortars at three weeks ago, and then service is terrible.
#     {'neg': 0.114, 'neu': 0.886, 'pos': 0.0, 'compound': -0.4767}
#     Okay, Peter, sorry to hear about that.
#     {'neg': 0.159, 'neu': 0.61, 'pos': 0.232, 'compound': 0.1531}
#     Hey, Peter, before we go on, do you mind just, uh, is there something going on with your microphone?
#     {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
#     I can't quite hear you.
#     {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
#     Is this any better?
#     {'neg': 0.0, 'neu': 0.508, 'pos': 0.492, 'compound': 0.4404}
#     Yeah, that's much better.
#     {'neg': 0.0, 'neu': 0.282, 'pos': 0.718, 'compound': 0.6249}
#     And sorry, what was, what was it that you said when you first first started speaking?
#     {'neg': 0.08, 'neu': 0.92, 'pos': 0.0, 'compound': -0.0772}
#     So I ordered a product from you guys three weeks ago and, uh, it's, it's currently on July 1st and I haven't received a provocative, again, three weeks to a full four weeks down line.
#     {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
#     This service is terrible.
#     {'neg': 0.508, 'neu': 0.492, 'pos': 0.0, 'compound': -0.4767}
#     Okay.
#     {'neg': 0.0, 'neu': 0.0, 'pos': 1.0, 'compound': 0.2263}
#     Well, what's your order id?
#     {'neg': 0.0, 'neu': 0.656, 'pos': 0.344, 'compound': 0.2732}
#     I'll, uh, I'll start looking into that for you.
#     {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
#     Six, nine, eight, seven five.
#     {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
#     Okay.
#     {'neg': 0.0, 'neu': 0.0, 'pos': 1.0, 'compound': 0.2263}
#     Thank you.
#     {'neg': 0.0, 'neu': 0.286, 'pos': 0.714, 'compound': 0.3612}
----------

import spacy

# Transcribe call 4 channel 2
call_4_channel_2_text = transcribe_audio("call_4_channel_2.wav")

# Create a spaCy language model instance
nlp = spacy.load("en_core_web_sm")

# Create a spaCy doc with call 4 channel 2 text
doc = nlp(call_4_channel_2_text)

# Check the type of doc
print(type(doc))
# output:
#     <class 'spacy.tokens.doc.Doc'>
-----

import spacy

# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")

# Create a spaCy doc with call 4 channel 2 text
doc = nlp(call_4_channel_2_text)

# Show tokens in doc
for token in doc:
    print(token.text, token.idx)
# output:
#     oh 0
#     hello 3
#     Daniel 9
#     my 16
#     name 19
#     is 24
#     Ann 27
#     and 31
#     I 35
#     've 36
#     recently 40
#     just 49
#     purchased 54
#     are 64
#     a 68
#     smartphone 70
#     from 81
#     you 86
#     and 90
#     I 94
#     'm 95
#     very 98
#     happy 103
#     with 109
#     the 114
#     product 118
#     ID 126
#     like 129
#     to 134
#     order 137
#     another 143
#     one 151
#     for 155
#     my 159
#     friend 162
#     who 169
#     lives 173
#     in 179
#     Sydney 182
#     and 189
#     have 193
#     it 198
#     delivered 201
#     I 211
#     'm 212
#     pretty 215
#     sure 222
#     it 227
#     's 229
#     model 232
#     315 238
#     I 242
#     can 244
#     check 248
#     that 254
#     for 259
#     you 263
#     and 267
#     I 271
#     'll 272
#     give 276
#     you 281
#     my 285
#     details 288
#     arm 296
#     if 300
#     you 303
#     would 307
#     like 313
#     to 318
#     take 321
#     my 326
#     details 329
#     and 337
#     I 341
#     I 343
#     will 345
#     also 350
#     give 355
#     you 360
#     the 364
#     address 368
#     thank 376
#     you 382
#     excellent 386
--------------

import spacy

# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")

# Create a spaCy doc with call 4 channel 2 text
doc = nlp(call_4_channel_2_text)

# Show sentences in doc
for sentence in doc.sents:
    print(sentence)
--------
import spacy

# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")

# Create a spaCy doc with call 4 channel 2 text
doc = nlp(call_4_channel_2_text)

# Show named entities and their labels
for entity in doc.ents:
    print(entity.text, entity.label_)
# output:
#     Ann PERSON
#     Sydney GPE
#     315 CARDINAL
# You've now seen some of spaCy's helpful functions for analyzing text. spaCy's built-in named entities are great to start with but sometimes you'll want to use your own. Let's see how!
------------

# Import EntityRuler class
from spacy.pipeline import EntityRuler

# Create EntityRuler instance
ruler = EntityRuler(nlp)

# Define pattern for new entity
ruler.add_patterns([{"label": "PRODUCT", "pattern": "smartphone"}])

# Update existing pipeline
nlp.add_pipe(ruler.add_patterns, before="ner")

# Test new entity
for entity in doc.ents:
  print(entity.text, entity.label_)
# With custom entities like this, you can start to get even more information out of your transcribed text. Depending on the problem you're working with, you may want to combine a few different patterns together. Let's keep going.
# output:
#     Ann PERSON
#     Sydney GPE
#     315 CARDINAL
--------
# Convert post purchase
for file in post_purchase:
    print(f"Converting {file} to .wav...")
    convert_to_wav("post_purchase_audio_0.mp3")

# Convert pre purchase
for file in pre_purchase:
    print(f"Converting {file} to .wav...")
    convert_to_wav(file)
# Now all of the audio files are in .wav format, let's transcribe them.
----------
def create_text_list(folder):
  # Create empty list
  text_list = []
  
  # Go through each file
  for file in folder:
    # Make sure the file is .wav
    if file.endswith(".wav"):
      print(f"Transcribing file: {file}...")
      
      # Transcribe audio and append text to list
      text_list.append(transcribe_audio(file))   
  return text_list

create_text_list(folder)
-------
# Transcribe post and pre purchase text
post_purchase_text = create_text_list(post_purchase_wav_files)
pre_purchase_text = create_text_list(pre_purchase_wav_files)

# Inspect the first transcription of post purchase
print(post_purchase_text[0])
# We've now got two lists of transcribed audio snippets we can use to start building a text classifier. Let's organize our text data a little bit with a dataframe.
print(post_purchase_text[0])

# <script.py> output:
#     Transcribing file: post_purchase_audio_0.wav...
#     Transcribing file: post_purchase_audio_1.wav...
#       .
#       .
#       .
#     Transcribing file: pre_purchase_audio_21.wav...
#     Transcribing file: pre_purchase_audio_22.wav...
    
#     hey man I just water product from you guys and I think is amazing but I leave a little help setting it up

------------
import pandas as pd

# Make dataframes with the text
post_purchase_df = pd.DataFrame({"label": "post_purchase",
                                 "text": post_purchase_text})
pre_purchase_df = pd.DataFrame({"label": "pre_purchase",
                                "text": pre_purchase_text})

# Combine DataFrames
df = pd.concat([post_purchase_df, pre_purchase_df])

# Print the combined DataFrame
print(df.head())
# output:
#                label                                               text
#     0  post_purchase  hey man I just bought a product from you guys ...
#     1  post_purchase  these clothes I just bought from you guys too ...
#     2  post_purchase  I recently got these pair of shoes but they're...
#     3  post_purchase  I bought a pair of pants from you guys but the...
#     4  post_purchase  I bought a pair of pants and they're the wrong...
# That was the final piece of the puzzle! Having your data in an organized format makes it easier to work with in the future. Let's go and build that text classifier.
------------

# Build the text_classifier as an sklearn pipeline
text_classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB()),
])

# Fit the classifier pipeline on the training data
text_classifier.fit(train_df.text, train_df.label)

# Evaluate the MultinomialNB model
predicted = text_classifier.predict(test_df.text)
accuracy = 100 * np.mean(predicted == test_df.label)
print(f'The model is {accuracy}% accurate')

# output:
#     The model is 90.47619047619048% accurate
# The model was able to classify our test examples with a high level of accuracy. For larger datasets, our pipeline is a good baseline but you might want to look into something like a language model. Now you can start capturing speech, converting it to text and classifying it into different categories. Massive effort!
------------


