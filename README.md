# musicgen
Creating a LSTM music generator using Keras.
Used tutorial from https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5

## Example results

(All included in the repo)

midi file: test_output.midi (use a midi reader like Musescore to open)

mp3 audio file: test-output.mp3

## Install dependencies
```
pip install -r requirements.txt
```
## To run the trained music generator
Make sure that you are running on python 3.6
```
python predict.py
```
## To train the model yourself
```
python music_gen_lstm.py
```
Replace ```weights-improvement-107-1.0217-bigger.hdf5``` in line 70 of ```predict.py``` with the last produced weights file of the neural net.
```
python predict.py
```
Now you have your generated MIDI! You can use an online MIDI->MP3 converter or a MIDI reader to listen to your generated music.

You can also replace the midi files in the training folder with your own music if you want a different style of music. I used Impressionism-era piano music, but anything works (single instrument files work best).
