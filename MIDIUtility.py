import magenta
from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.models.shared import sequence_generator_bundle
from magenta.music import midi_io
from magenta.music.protobuf import generator_pb2
from magenta.music.protobuf import music_pb2
import pretty_midi
from music21 import midi
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pygame

from MIDIHyperparameters import CLIP_LENGTH


def visualizePianoRoll(outfile, title_label):
    midi_data = pretty_midi.PrettyMIDI(outfile)
    plt.figure(figsize=(12, 6))

    high_note = 50
    low_note = 50
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            # Note start, note pitch, note duration, note velocity
            start = note.start
            pitch = note.pitch
            duration = note.end - note.start
            velocity = note.velocity

            if high_note < pitch:
                high_note = pitch
            if low_note > pitch:
                low_note = pitch

            # Draw a rectangle for each note
            rect = patches.Rectangle((start, pitch), duration, 1,
                                     edgecolor='black', facecolor='orangered', alpha=0.5)
            plt.gca().add_patch(rect)

    # Set the y-axis to cover the range of MIDI notes
    plt.ylim(low_note - 2, high_note + 2)
    # Set the x-axis to cover the time range of the MIDI file
    plt.xlim(0, midi_data.get_end_time())
    plt.xlabel('Time (s)')
    plt.ylabel('MIDI Note Number')
    plt.title('Piano Roll: ' + str(title_label))

    # Display the piano roll
    plt.savefig(outfile[:-4] + '_plot.png')


def initializeModel(model_type='attention_rnn', num_seconds=CLIP_LENGTH):
    model = None
    if model_type == 'attention_rnn':
        bundle = sequence_generator_bundle.read_bundle_file('attention_rnn.mag')
        generator_map = melody_rnn_sequence_generator.get_generator_map()
        melody_rnn = generator_map['attention_rnn'](checkpoint=None, bundle=bundle)
        melody_rnn.initialize()
        model = melody_rnn

    input_sequence = music_pb2.NoteSequence()  # Empty NoteSequence
    generator_options = generator_pb2.GeneratorOptions()  # Generator options

    # Here we specify how many seconds of music to generate.
    generator_options.generate_sections.add(start_time=0, end_time=num_seconds)
    return model, input_sequence, generator_options


def generateMidiPerformance(model, input_sequence, generator_options, outfile):
    # Generate the sequence with the generator options and input sequence
    sequence = model.generate(input_sequence, generator_options)

    # Save the generated sequence to a MIDI file
    midi_io.note_sequence_to_midi_file(sequence, outfile)
    return sequence


def playMidiFile(outfile):
    # Load and play the MIDI file
    pygame.mixer.music.load(outfile)
    print('Playing generated MIDI file...')
    pygame.mixer.music.play()

    # Keep the script running until the music is done playing
    while pygame.mixer.music.get_busy():
        pygame.time.wait(1000)


def extractFeatures(sequence, index):
    # Initialize feature set
    features = []

    # Current note features
    current_note = sequence.notes[index]
    features.extend([current_note.pitch, current_note.end_time - current_note.start_time, current_note.start_time])

    return features


# based on the sequence generated, reward the model for following the sequence.
def convertMidiToStates(sequence):
    sequence_states = []

    for i in range(len(sequence.notes)):
        features = extractFeatures(sequence, i)
        sequence_states.append(features)


    for i, state in enumerate(sequence_states):
        sequence_states[i] = [state[0] - 48, int(state[1] * 8) - 1, int(state[2] * 8)]

    return sequence_states


def convertStatesToMidi(state_sequence):
    midi = pretty_midi.PrettyMIDI()

    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)

    for i, state in enumerate(state_sequence):
        pitch = state[0]
        if (state[0] != 0):
            pitch += 48
        duration = (state[1] + 1) * 0.125
        start = state[2] / 8
        end = start + duration
        note = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=start,
            end=end
        )
        piano.notes.append(note)

    midi.instruments.append(piano)
    return midi


def getKeyModifier(key='A'):
    keyMap = { 'A':8, 'Bb':9, 'A#':9, 'B':10, 'C':11, 'Db':0, 'C#':0,
               'D':1, 'Eb':2, 'D#':2, 'E':3, 'F':4, 'Gb':5, 'F#':5,
               'G':6, 'Ab':7, 'G#':7 }
    return keyMap[key]


def getScaleDegrees(scale='ionian'):
    modeMap = { 'ionian'     : [0, 2, 4, 5, 7, 9, 11],
                'dorian'     : [0, 2, 3, 5, 7, 9, 10],
                'phrygian'   : [0, 1, 3, 5, 7, 8, 10],
                'lydian'     : [0, 2, 4, 6, 7, 9, 11],
                'mixolydian' : [0, 2, 4, 5, 7, 9, 10],
                'aeolian'    : [0, 2, 3, 5, 7, 8, 10],
                'locrian'    : [0, 1, 3, 5, 6, 8, 10] }
    return modeMap[scale]


def generateScaleSequence(key='A', scale='ionian'):
    state_sequence = []

    pitch = getKeyModifier(key)
    duration = 2
    time = 1
    degrees = getScaleDegrees(scale)
    octaves = 3

    for octave in range(octaves):
        for degree in degrees:
            state_sequence.append([pitch + degree + (octave * 12) + 1, duration, time])
            time += duration + 1

    return state_sequence

