# Reinforcement Learning for Music Generation


Implemented and Developed by Emmet Hayes


This project was created as a final project for a Masters-level course in Reinforcement Learning. It explores the effectiveness of off-policy Monte Carlo control methods applied to the task of music generation. In order to run the code, this project requires a working version of Magenta, by Google, at this link: https://github.com/magenta/magenta


Follow the installation steps that are included on the Magenta repository link above. Additionally, there are a few more required dependencies to run the code:


pygame, imageio, matplotlib, and seaborn


To run the code, run the following command in the working directory of the project:


`python3 MIDIMain.py --zero-state`


Here is a list of arguments available for modifying the process of training:


--zero-state : Resets the saved MIDI state.

--dont-play : Skip playing the MIDI output during run.

--scale : Accepts a type of scale as the target sequence (major, minor, ionian, dorian, phrygian, lydian, mixolydian, aeolian, locrian)

--key : Accepts a musical key for the target sequence to be in.

--human-feedback : Enables human feedbackm which will periodically ask the user to rate the performance based on three subjective metrics.


There are two branches for this project that were used for evaluating and completing the project. The main branch is dedicated to monophonic melody sequence generation, while the polyphonic branch is modified to handle polyphonic sequence generation.
