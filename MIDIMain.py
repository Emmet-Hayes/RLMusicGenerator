import numpy as np
import pygame
import argparse
from magenta.music import midi_io

from MIDIEnvironment import MIDIEnvironment
from MIDIAgent import MIDIAgent
from MIDIHyperparameters import NUM_EPISODES, PITCH_COUNT, DURATION_COUNT, CLIP_LENGTH
from MIDIUtility import visualizePianoRoll, initializeModel, generateMidiPerformance, playMidiFile, convertMidiToStates, convertStatesToMidi, generateScaleSequence


# Argument parsing
parser = argparse.ArgumentParser(description="Control behavior of the music generation agent.")
parser.add_argument("--zero-state", action="store_true", help="Reset saved MIDI state.")
parser.add_argument("--dont-play", action="store_true", help="Skip playing the MIDI output")
parser.add_argument("--scale", type=str, help="Accepts a type of scale as the target sequence (ionian, dorian, phrygian, lydian, mixolydian, aeolian, locrian")
parser.add_argument("--key", type=str, help="Accepts a musical key for the target sequence to be in.")
parser.add_argument("--q-learning", action="store_true", help="Uses q-learning instead of monte carlo")
args = parser.parse_args()

pygame.mixer.init()

# model, input_sequence, generator_options = initializeModel('attention_rnn')
outfile = 'melody_1.mid'

if args.zero_state:
    if args.scale or args.key:
        scale = 'ionian'
        if args.scale:
            scale = args.scale

        key = 'A'
        if args.key:
            key = args.key

        sequence_states = generateScaleSequence(key, scale)
        midi = convertStatesToMidi(sequence_states)
        midi.write(outfile)
    else:
        model, input_sequence, generator_options = initializeModel()
        sequence = generateMidiPerformance(model, input_sequence, generator_options, outfile)
else:
    sequence = midi_io.midi_file_to_note_sequence(outfile)

if not args.dont_play:
    playMidiFile(outfile)
visualizePianoRoll(outfile)

if not args.scale:
    sequence_states = convertMidiToStates(sequence)


# this way, our state space IS almost essentially the same as our action space.
# only the environment will be aware of how much time has passed.
env = MIDIEnvironment(sequence_states)

# zero the state from previous runs
if args.zero_state:
    env.Qvalues = np.random.rand(PITCH_COUNT, DURATION_COUNT, 8 * CLIP_LENGTH, PITCH_COUNT * DURATION_COUNT) * 400 - 500
    env.rewards = []
    env.Cvalues = np.zeros((PITCH_COUNT, DURATION_COUNT, 8 * CLIP_LENGTH, 37*8))
    env.policy = np.zeros((PITCH_COUNT, DURATION_COUNT, 8 * CLIP_LENGTH), dtype = 'int')

    env.saveQvalues()
    env.saveCvalues()
    env.savePolicy()
    env.saveRewards()

agent = MIDIAgent(env)

for i in range(NUM_EPISODES):
    if args.q_learning:
        agent.qControl()
    else:
        agent.mcControl()

    if (i + 1) % 10 == 0:
        agent.evaluateTargetPolicy()

    if (i + 1) % 1000 == 0:
        print("On Episode #" + str(i + 1))

    if (i + 1) % 100000 == 0 or i == 100:
        agent.saveTrackData()
        agent.plotRewards('Performance')
        agent.plotQValueHeatmap(time_step=40, episode=(i + 1))
        agent.plotPolicy(time_step=40, episode=(i + 1))
        agent.plotActionHistogram(episode=(i + 1))
        agent.plotRewardComponentBreakdown(episode=(i + 1))
        # now that the RL model is trained, we should track its state transitions and
        # parse them back into a MIDI sequence.
        final_states = agent.outputStatesFromTargetPolicyRun()

        midi = convertStatesToMidi(final_states)

        final_outfile = 'final_output_' + str(i + 1) + '.mid'
        midi.write(final_outfile)

        if not args.dont_play:
            playMidiFile(final_outfile)
        visualizePianoRoll(final_outfile, i)

