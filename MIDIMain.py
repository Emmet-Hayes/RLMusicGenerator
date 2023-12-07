import numpy as np
import pygame
import argparse
import imageio
from magenta.music import midi_io

from MIDIEnvironment import MIDIEnvironment
from MIDIAgent import MIDIAgent
from MIDIHyperparameters import NUM_EPISODES, PITCH_COUNT, DURATION_COUNT, CLIP_LENGTH, PLOT_FREQUENCY
from MIDIUtility import visualizePianoRoll, initializeModel, generateMidiPerformance, playMidiFile, convertMidiToStates, convertStatesToMidi, generateScaleSequence
from MIDIDQNAgent import DQNAgent

# Argument parsing
parser = argparse.ArgumentParser(description="Control behavior of the music generation agent.")
parser.add_argument("--zero-state", action="store_true", help="Reset saved MIDI state.")
parser.add_argument("--dont-play", action="store_true", help="Skip playing the MIDI output")
parser.add_argument("--scale", type=str, help="Accepts a type of scale as the target sequence (ionian, dorian, phrygian, lydian, mixolydian, aeolian, locrian")
parser.add_argument("--key", type=str, help="Accepts a musical key for the target sequence to be in.")
parser.add_argument("--q-learning", action="store_true", help="Uses q-learning instead of DQN")
parser.add_argument("--monte-carlo", action="store_true", help="Uses monte carlo instead of DQN")
parser.add_argument("--human-feedback", action="store_true", help="Asks for human feedback based on the authenticity of the performance")
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
visualizePianoRoll(outfile, 'Performance (baseline)')


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
    elif args.monte_carlo:
        agent.mcControl()
    else:
        agent.dqnControl()
        if i > NUM_EPISODES // 10:
            break

    if (i + 1) % 10 == 0:
        print("On Episode #" + str(i + 1))
        agent.evaluateTargetPolicy()


    if (i + 1) % 1000 == 0:
        print("On Episode #" + str(i + 1))

    if (i + 1) % 10000 == 0:
        agent.saveTrackData()

    if args.human_feedback and (i + 1) % PLOT_FREQUENCY * 3 == 0:
        print("What would you rate the creativity of the song? (1-10) ")
        rating = input()
        agent.env.correct_key_reward_hf_mod = 5 - int(rating)

        print("What would you rate the naturalness of the song? (1-10) ")
        rating = input()
        agent.env.correct_timing_reward_hf_mod = 5 - int(rating)

        print("What would you rate the authenticity of the song? (1-10) ")
        rating = input()
        agent.env.correct_note_reward_hf_mod = 5 - int(rating)

    if (i + 1) % PLOT_FREQUENCY == 0 or (i + 1) == 100:
        if args.q_learning or args.monte_carlo:
            agent.plotRewards('Performance')
            agent.plotQValueHeatmap(time_step=40, episode=(i + 1))
            agent.plotPolicy(time_step=40, episode=(i + 1))
            agent.plotActionHistogram(episode=(i + 1))
            agent.plotRewardComponentBreakdown(episode=(i + 1))
        else:
            agent.saveModel()
        # now that the RL model is trained, we should track its state transitions and
        # parse them back into a MIDI sequence.
        final_states = agent.outputStatesFromTargetPolicyRun()

        midi = convertStatesToMidi(final_states)

        final_outfile = 'final_output_' + str(i + 1) + '.mid'
        midi.write(final_outfile)

        if not args.dont_play:
            playMidiFile(final_outfile)
        visualizePianoRoll(final_outfile, 'Performance Episode #' + str(i + 1))


# Generate images and videos showing model progress
ah_filenames = [] # Action Histograms
fop_filenames = [] # Final Output Plots
pp_filenames = [] # Policy Plots
qvh_filenames = [] # Q-value Heatmaps
rb_filenames = [] # Reward Breakdown

for i in range(NUM_EPISODES // PLOT_FREQUENCY):
    if i == 0:
        ah_filenames.append('Action_Histogram_episode100.png')
        fop_filenames.append('final_output_100_plot.png')
        pp_filenames.append('Policy_Plot_100_40.png')
        qvh_filenames.append('QValue_Heatmap_episode_100_0_40.png')
        rb_filenames.append('Reward_Breakdown_episode100.png')
    else:
        ah_filenames.append('Action_Histogram_episode' + str(i * PLOT_FREQUENCY) + '.png')
        fop_filenames.append('final_output_' + str(i * PLOT_FREQUENCY) + '_plot.png')
        pp_filenames.append('Policy_Plot_' + str(i * PLOT_FREQUENCY) + '_40.png')
        qvh_filenames.append('QValue_Heatmap_episode_' + str(i * PLOT_FREQUENCY) + '_0_40.png')
        rb_filenames.append('Reward_Breakdown_episode' + str(i * PLOT_FREQUENCY) + '.png')

ah_images = [imageio.imread(filename) for filename in ah_filenames]
fop_images = [imageio.imread(filename) for filename in fop_filenames]
pp_images = [imageio.imread(filename) for filename in pp_filenames]
qvh_images = [imageio.imread(filename) for filename in qvh_filenames]
rb_images = [imageio.imread(filename) for filename in rb_filenames]


imageio.mimsave('ah_movie.gif', ah_images, duration = 0.2)
imageio.mimsave('fop_movie.gif', fop_images, duration = 0.2)
imageio.mimsave('pp_movie.gif', pp_images, duration = 0.2)
imageio.mimsave('qvh_movie.gif', qvh_images, duration = 0.2)
imageio.mimsave('rb_movie.gif', rb_images, duration = 0.2)
