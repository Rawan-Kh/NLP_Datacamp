#  All of the Recognizer class API calls begin with recognize_.
--------
# Importing the speech_recognition library
import speech_recognition as sr

# Create an instance of the Recognizer class
recognizer = sr.Recognizer()

# Set the energy threshold
recognizer.energy_threshold = 300
--------
# Create a recognizer class
recognizer = sr.Recognizer()

# Transcribe the support call audio
text = recognizer.recognize_google(
  audio_data=clean_support_call_audio, 
  language="en-US")

print(text)
# output:
#     hello I'd like to get some help setting up my account please
# You just transcribed your first piece of audio using speech_recognition's Recognizer class! Well, we've set it a mock version of Recognizer so we don't hit the API max requests limit. Notice how the 'hello' wasn't seperate from the rest of the text. As powerful as recognize_google() is, it doesn't have sentence separation.
-----------
# Instantiate Recognizer
recognizer = sr.Recognizer()

# Convert audio to AudioFile
clean_support_call = sr.AudioFile('clean_support_call.wav')

# Convert AudioFile to AudioData
with clean_support_call as source:
    clean_support_call_audio = recognizer.record(source)

# Transcribe AudioData to text
text = recognizer.recognize_google(clean_support_call_audio,
                                   language="en-US")
print(text)

# You've gone end to end with SpeechRecognition, you've imported an audio file, converted it to the right data type and transcribed it using Google's free web API! Now let's see a few more capabilities of the record() method.
# output:
#     hello I'd like to get some help setting up my account please
-------------------------

# Convert AudioFile to AudioData
with static_at_start as source:
    static_art_start_audio = recognizer.record(source,
                                               duration=None,
                                               offset=3)

# Transcribe AudioData to text
text = recognizer.recognize_google(static_art_start_audio,
                                   language="en-US")

print(text)
# Speech recognition can be resource intensive, so in practice, you'll want to explore your audio files to make you're not wasting any compute power trying to transcribe static or silence.
#  output:
#     hello ID like to get some help with my device please I think it's out of warranty I bought it about two years ago
----------------

# Create a recognizer class
recognizer = sr.Recognizer()

# Pass the Japanese audio to recognize_google
text = recognizer.recognize_google(japanese_audio, language="en-US")

# Print the text
print(text)
# ohayo gozaimasu
---------------

# Create a recognizer class
recognizer = sr.Recognizer()

# Pass the Japanese audio to recognize_google
text = recognizer.recognize_google(japanese_audio, language="ja")

# Print the text
print(text)
# おはようございます
--------------
# Create a recognizer class
recognizer = sr.Recognizer()

# Pass the leopard roar audio to recognize_google
text = recognizer.recognize_google(leopard_audio, 
                                   language="en-US", 
                                   show_all=True)

# Print the text
print(text)
# output:
#     []
----------
# Create a recognizer class
recognizer = sr.Recognizer()

# Pass charlie_audio to recognize_google
text = recognizer.recognize_google(charlie_audio, 
                                   language="en-US")

# Print the text
print(text)
# output:
#     Charlie Charlie bit me
# You've seen how the recognize_google() deals with different kinds of audio. It's worth noting the recognize_google() function is only going to return words, as in, it didn't return the baby saying 'ahhh!' because it doesn't recognize it as a word. Speech recognition has come a long way but it's far from perfect. Let's push on!
--------------

# Create a recognizer class
recognizer = sr.Recognizer()

# Recognize the multiple speaker AudioData
text = recognizer.recognize_google(multiple_speakers, 
                       			   language="en-US")

# Print the text
print(text)
# output:
#     one of the limitations of the speech recognition library is that it doesn't recognise different speakers and voices it will just return it all as one block a text
# You did it. But see how all of the speakers speech came out in one big block of text? In the next exercise we'll see a way of working around this.
---------

recognizer = sr.Recognizer()

# Multiple speakers on different files
speakers = [sr.AudioFile("speaker_0.wav"), 
            sr.AudioFile("speaker_1.wav"), 
            sr.AudioFile("speaker_2.wav")]

# Transcribe each speaker individually
for i, speaker in enumerate(speakers):
    with speaker as source:
    # Call record() on recognizer to convert the AudioFiles into AudioData.
        speaker_audio = recognizer.record(source)
    print(f"Text from speaker {i}:")
    # Use recognize_google() to transcribe each of the speaker_audio objects.
    print(recognizer.recognize_google(speaker_audio,
         				  language="en-US"))
    
# output:
#     Text from speaker 0:
#     one of the limitations of the speech recognition library
    
#     Text from speaker 1:
#     is that it doesn't recognise different speakers and voices
    
#     Text from speaker 2:
#     it will just return it all as one block of text

#Something to remember is I had to manually split the audio file into different speakers. You can see this solution still isn't perfect but it's easier to deal with than having a single block of text. You could think about automating this process in the future by having a model split the audio when it detects different speakers. For now, let's look into what happens when you've got noisy audio!
-----------
recognizer = sr.Recognizer()

# Record the audio from the clean support call
with clean_support_call as source:
  clean_support_call_audio = recognizer.record(source)

# Transcribe the speech from the clean support call
text = recognizer.recognize_google(clean_support_call,
					   language="en-US")

print(text)
----------------
recognizer = sr.Recognizer()

# Record the audio from the noisy support call
with noisy_support_call as source:
  noisy_support_call_audio = recognizer.record(source)

# Transcribe the speech from the noisy support call
text = recognizer.recognize_google(noisy_support_call_audio,
                         language="en-US",
                         show_all=True)

print(text)
# output:
#     {'alternative': [{'transcript': 'hello ID like to get some help setting up my calories', 'confidence': 0.75329071}, {'transcript': 'hello ID like to get some colour setting on my account please'}, {'transcript': 'hello ID like to get some colour setting on my calendar'}, {'transcript': 'hello ID like to get some help setting up my account please'}, {'transcript': 'hello ID like to get some colour setting on my account'}], 'final': True}
------------

recognizer = sr.Recognizer()

# Record the audio from the noisy support call
with noisy_support_call as source:
	# Adjust the recognizer energy threshold for ambient noise
    recognizer.adjust_for_ambient_noise(source, duration=1)
    noisy_support_call_audio = recognizer.record(noisy_support_call)
 
# Transcribe the speech from the noisy support call
text = recognizer.recognize_google(noisy_support_call_audio,
                                   language="en-US",
                                   show_all=True)

print(text)
---------
recognizer = sr.Recognizer()

# Record the audio from the noisy support call
with noisy_support_call as source:
	# Adjust the recognizer energy threshold for ambient noise
    recognizer.adjust_for_ambient_noise(source, duration=0.5)
    noisy_support_call_audio = recognizer.record(noisy_support_call)
 
# Transcribe the speech from the noisy support call
text = recognizer.recognize_google(noisy_support_call_audio,
                                   language="en-US",
                                   show_all=True)

print(text)
# output:
#     {'alternative': [{'transcript': 'hello ID like to get some help setting up my calories', 'confidence': 0.88225818}, {'transcript': 'hello ID like to get some help setting up my account please'}, {'transcript': 'hello ID like to get some help setting up my account'}, {'transcript': 'hello ID like to get some help setting on my account please'}, {'transcript': 'hello ID like to get some help setting up my Kelly please'}], 'final': True}
------------
# the results still weren't perfect. This should be expected with some audio files though, sometimes the background noise is too much. If your audio files have a large amount of background noise, you may need to preprocess them with an audio tool such as Audacity before using them with speech_recognition.
--------
