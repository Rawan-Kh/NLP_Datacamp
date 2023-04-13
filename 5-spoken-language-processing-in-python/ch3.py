# Import AudioSegment from Pydub
from pydub import AudioSegment

# Create an AudioSegment instance
wav_file = AudioSegment.from_file(file='wav_file.wav', 
                                  format="wav")

# Check the type
print(type(wav_file))
# You've just imported your first audio file using PyDub. Over the next few lessons, you'll start to see how many helpful functions PyDub has built-in for working with audio.
-----

# Import AudioSegment and play
from pydub import AudioSegment
from pydub.playback import play

# Create an AudioSegment instance
wav_file = AudioSegment.from_file(file="wav_file.wav", 
                                  format="wav")

# Play the audio file
play(wav_file)
---------
# Import audio file
wav_file = AudioSegment.from_file(file="wav_file.wav")

# Find the frame rate
print(wav_file.frame_rate)
------
# Import audio file
wav_file = AudioSegment.from_file(file="wav_file.wav")

# Find the number of channels
print(wav_file.channels)
---------
# Import audio file
wav_file = AudioSegment.from_file(file="wav_file.wav")

# Find the max amplitude
print(wav_file.max)
--------
# Import audio file
wav_file = AudioSegment.from_file(file="wav_file.wav")

# Find the length
print(len(wav_file))
# Massive effort! There are many more characteristics you can find out about your audio files once you've imported them as an AudioSegment. Try find some more by adding a dot after your audio file (wav_file.) and pressing tab.
--------
# Import audio file
wav_file = AudioSegment.from_file(file="wav_file.wav")

# Create a new wav file with adjusted frame rate
wav_file_16k = wav_file.set_frame_rate(16000)

# Check the frame rate of the new wav file
print(wav_file_16k.frame_rate)

------------
# Import audio file
wav_file = AudioSegment.from_file(file="wav_file.wav")

# Set number of channels to 1
wav_file_1_ch = wav_file.set_channels(1)

# Check the number of channels
print(wav_file_1_ch.channels)
----------
# Import audio file
wav_file = AudioSegment.from_file(file="wav_file.wav")

# Print sample_width
print(f"Old sample width: {wav_file.sample_width}")

# Set sample_width to 1
wav_file_sw_1 = wav_file.set_sample_width(1)

# Check new sample_width
print(f"New sample width: {wav_file_sw_1.sample_width}")
# output:
#     Old sample width: 2
#     New sample width: 1
# Once again, there are other methods you can call on your AudioSegment instances to adjust their attributes as further practice, you should try and find some more. But remember, lowering the values generally leads to lower audio qaulity and worse transcriptions but increasing them may increase the file size and but not the quality of the transcription. Best to explore with different values and find out the ideal tradeoff.
--------

from pydub import AudioSegment

# Import audio file
volume_adjusted = AudioSegment.from_file('volume_adjusted.wav')

# Lower the volume by 60 dB
quiet_volume_adjusted = volume_adjusted - 60
----------
from pydub import AudioSegment

# Import audio file
volume_adjusted = AudioSegment.from_file('volume_adjusted.wav')

# Increase the volume by 15 dB
louder_volume_adjusted = volume_adjusted + 15
# That sounds like progress! Here's the louder audio file you created and the quieter one (no sound). Nice work! Adjusting the volume with operators can be useful but doesn't help when you only want to increase the loudness of only quiet sections. Let's take a look at a function which can help!
-----------

# Import AudioSegment and normalize
from pydub import AudioSegment
from pydub.effects import normalize

# Import target audio file
loud_then_quiet = AudioSegment.from_file('loud_then_quiet.wav')

# Normalize target audio file
normalized_loud_then_quiet = normalize(loud_then_quiet)
# Remember, speech recognition works best on clear speech files, so the more you can do to improve the quality of your audio files, including their volume, the better.
--------
from pydub import AudioSegment

# Import part 1 and part 2 audio files
part_1 = AudioSegment.from_file('part_1.wav')
part_2 = AudioSegment.from_file('part_2.wav')

# Remove the first four seconds of part 1
part_1_removed = part_1[4000:]

# Add the remainder of part 1 and part 2 together
part_3 = part_1_removed + part_2
# You're becoming an audio manipulation master! But we're not done yet, there's still a few more tricks in the PyDub library you should know about.
--------

# Import AudioSegment
from pydub import AudioSegment

# Import stereo audio file and check channels
stereo_phone_call = AudioSegment.from_file("stereo_phone_call.wav")
print(f"Stereo number channels: {stereo_phone_call.channels}")

# Split stereo phone call and check channels
channels = stereo_phone_call.split_to_mono()
print(f"Split number channels: {channels[0].channels}, {channels[1].channels}")

# Save new channels separately
phone_call_channel_1 = channels[0]
phone_call_channel_2 = channels[1]
#  Having audio files with only one speaker usually results in better quality transcriptions. Now you've done all this audio processing, how do save your altered audio files to use later? Let's find out.
-----------

from pydub import AudioSegment

# Import the .mp3 file
mp3_file = AudioSegment.from_file('mp3_file.mp3')

# Export the .mp3 file as wav
mp3_file.export(out_f='mp3_file.wav',
                format='wav')
# Now our .mp3 file is in the .wav format, it'll definitely be compatible with all kinds of speech transcription APIs. Let's see this at scale.
---------

# Loop through the files in the folder
for audio_file in folder:
    
	# Create the new .wav filename
    wav_filename = os.path.splitext(os.path.basename(audio_file))[0] + ".wav"
        
    # Read audio_file and export it in wav format
    AudioSegment.from_file(audio_file).export(out_f=wav_filename, 
                                      format='wav')
        
    print(f"Creating {wav_filename}...")

# <script.py> output:
#     Creating good_afternoon_mp3.wav...
#     Creating good_afternoon_m4a.wav...
#     Creating good_afternoon_aac.wav...
# You've successfully converted the folder of audio files from being non-compatiable with speech_recognition to being compatible!
----------
for audio_file in folder:
    file_with_static = AudioSegment.from_file(audio_file)

    # Cut the 3-seconds of static off
    file_without_static = file_with_static[3000:]

    # Increase the volume by 10dB
    louder_file_without_static = file_without_static + 10
    
    # Create the .wav filename for export
    wav_filename = os.path.splitext(os.path.basename(audio_file))[0] + ".wav"
    
    # Export the louder file without static as .wav
    louder_file_without_static.export(wav_filename, format='wav')
    print(f"Creating {wav_filename}...")

# ou've successfully processed and converted the folder of audio files from being non-compatiable with speech_recognition to being compatible! Here's what your files sound like without static, and here's without the static and 10 decibels louder. Let's start putting all you've learned about audio processing to work in the next chapter.
-------------
