from music21 import converter, instrument, note, chord
import glob
import pickle
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


def get_notes():

    notes = []

    """parsing files"""

    for file in glob.glob("training/*.midi"):
        midi = converter.parse(file)
        print("Parsing %s" % file)

        notes_to_parse = None

        parts = instrument.partitionByInstrument(midi)

        if parts: #file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else: #file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note): #if it is a note
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord): # if it is a chord
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, n_vocab):

    """mapping note strings to numbers"""

    seq_len = 100

    #get all pitch names
    pitchnames = sorted(set(item for item in notes))

    #create a dictionary to map pitches to integers
    note_to_int = dict((note,number) for number,note in enumerate(pitchnames))
    net_in = []
    net_out = []

    #create inputs and outputs
    for i in range(0, len(notes) - seq_len, 1):
        seq_in = notes[i:i + seq_len]
        seq_out = notes[i + seq_len]
        net_in.append([note_to_int[char] for char in seq_in])
        net_out.append(note_to_int[seq_out])

    n_patterns = len(net_in)

    #reshape input into LSTM-compatible format
    net_in = numpy.reshape(net_in, (n_patterns, seq_len, 1))
    #normalize input
    net_in = net_in/float(n_vocab)

    net_out = np_utils.to_categorical(net_out)

    return (net_in, net_out)

def create_network(net_in, n_vocab):
    """create the structure of neural net"""
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(net_in.shape[1], net_in.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train(model, net_in, net_out):
    """train the neural net!"""

    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint (
        filepath,
        monitor='loss',
        verbose = 0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(net_in, net_out, epochs=20, batch_size=64, callbacks=callbacks_list)


def train_network():
    '''trains the neural net'''
    notes = get_notes() #get notes
    n_vocab = len(set(notes)) #get number of pitch names
    net_in, net_out = prepare_sequences(notes, n_vocab)

    model = create_network(net_in, n_vocab)

    train(model, net_in, net_out)

if __name__ == '__main__':
    train_network()
