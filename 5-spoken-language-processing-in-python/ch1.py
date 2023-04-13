# Our audio files have been 48 kHz so far, that's 48,000 cycles per second!
-------------
import wave

# Create audio file wave object
good_morning = wave.open('good_morning.wav', 'r')

# Read all frames from wave object 
signal_gm = good_morning.readframes(-1)

# View first 10
print(signal_gm[:10])
# output:
#     b'\xfd\xff\xfb\xff\xf8\xff\xf8\xff\xf7\xff'
#  You've just imported your first audio file and seen what it looks like with pure Python. Now let's convert it something more readable.
-------------
# The 'int16' data type returns positive and negative integers, just as we wanted. Our sound wave integers will be ready to plot in no time!
----------------
import numpy as np

# Open good morning sound wave and read frames as bytes
good_morning = wave.open('good_morning.wav', 'r')
signal_gm = good_morning.readframes(-1)

# Convert good morning audio bytes to integers
soundwave_gm = np.frombuffer(signal_gm, dtype='int16')

# View the first 10 sound wave values
print(soundwave_gm[:10])
# output:
#     [ -3  -5  -8  -8  -9 -13  -8 -10  -9 -11]
# You've read in an audio file with Python and converted into to integers, that's the first step towards speech recognition!
----------
# Read in sound wave and convert from bytes to integers
good_morning = wave.open('good_morning.wav', 'r')
signal_gm = good_morning.readframes(-1)
soundwave_gm = np.frombuffer(signal_gm, dtype='int16')

# Get the sound wave frame rate
framerate_gm = good_morning.getframerate()

# Find the sound wave timestamps
time_gm = np.linspace(start=0,
                      stop=len(soundwave_gm)/framerate_gm,
					  num=len(soundwave_gm))

# Print the first 10 timestamps
print(time_gm[:10])
# You've done some great data manipulation to our good morning sound wave. Now we'll use all your hard work to plot it and see what it looks like!
-------------

#  Performing the same transformations on all of audio files, allows us to work with them in a consistent manner.
---------
# Setup the title and axis titles
plt.title('Good Afternoon vs. Good ____')
plt.ylabel('Amplitude')
plt.xlabel('Time (seconds)')

# Add the Good Afternoon data to the plot
plt.plot(time_ga, soundwave_ga, label='Good Afternoon')

# Add the Good Morning data to the plot
plt.plot(time_gm, soundwave_gm, label='Good Morning',
   # Set the alpha variable to 0.5
   alpha=0.5)

plt.legend()
plt.show()
# Notice the two sound waves are very similar in the beginning. Because the first word is good in both audio files, they almost completely overlap. A well-built speech recognition system would recognize this and return the same first word for each wave. Let's build one to do just that.
--------
