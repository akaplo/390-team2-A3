import os
import sys
import numpy as np
import math
from scipy.signal import lfilter
from scikits.talkbox import lpc

from aubio import pitch
from python_speech_features import mfcc

data_dir = 'data'
class_names = []
data = np.zeros((0,8002)) #8002 = 1 (timestamp) + 8000 (for 8kHz audio data) + 1 (label)

for filename in os.listdir(data_dir):
    if filename.endswith(".csv") and filename.startswith("speaker-data"):
        filename_components = filename.split("-") # split by the '-' character
        speaker = filename_components[2]
        print("Loading data for {}.".format(speaker))
        if speaker not in class_names:
            class_names.append(speaker)
        speaker_label = class_names.index(speaker)
        sys.stdout.flush()
        data_file = os.path.join('data', filename)
        data_for_current_speaker = np.genfromtxt(data_file, delimiter=',')
        print("Loaded {} raw labelled audio data samples.".format(len(data_for_current_speaker)))
        sys.stdout.flush()
        data = np.append(data, data_for_current_speaker, axis=0)

def compute_formants(audio_buffer):
    N = len(audio_buffer)
    Fs = 8000 # sampling frequency
    hamming_window = np.hamming(N)
    window = audio_buffer * hamming_window

    # Apply a pre-emphasis filter; this amplifies high-frequency components and attenuates low-frequency components.
    # The purpose in voice processing is to remove noise.
    filtered_buffer = lfilter([1], [1., 0.63], window)

    ncoeff = 2 + Fs / 1000
    A, e, k = lpc(filtered_buffer, ncoeff)
    roots = np.roots(A)
    roots = [r for r in roots if np.imag(r) >= 0]

    angz = np.arctan2(np.imag(roots), np.real(roots))

    unsorted_freqs = angz * (Fs / (2 * math.pi))

    freqs = sorted(unsorted_freqs)

    # also get the indices so that we can get the bandwidths in the same order
    indices = np.argsort(unsorted_freqs)
    sorted_roots = np.asarray(roots)[indices]

    #compute the bandwidths of each formant
    bandwidths = -1/2. * (Fs/(2*math.pi))*np.log(np.abs(sorted_roots))

    return freqs, bandwidths

def compute_formant_features(window):

    # Get the frequencies and bandwidths from the formants
    freqs, bandwidths = compute_formants(window)

    # Make a histogram with the default number of bins from the frequency data
    hist = np.histogram(freqs, bins=10)[0]

    # Return the histogram
    return hist

def compute_pitch_contour(window):
    win_s = 4096 # fft size
    hop_s = 512  # hop size
    samplerate=8000
    tolerance=0.8
    pitch_o = pitch("yin", win_s, hop_s, samplerate)
    pitch_o.set_unit("midi")
    pitch_o.set_tolerance(tolerance)

    INT_TO_FLOAT = 1. / 32768. #2^15 (15 not 16 because it's signed)


    # convert to an array of 32-bit floats in the range [-1,1]
    pitch_input = np.float32(window * INT_TO_FLOAT)

    pitch_contour = []
    confidence_curve = []

    index = 0
    while True:
        samples = pitch_input[index*hop_s:(index+1)*hop_s]
        pitch_output = pitch_o(samples)[0]
        confidence = pitch_o.get_confidence()
        pitch_contour += [pitch_output] # append to contour
        confidence_curve += [confidence]
        index += 1
        if (index+1)*hop_s > len(pitch_input): break # stop when there are no more frames

    return pitch_contour, confidence_curve

def compute_pitch_features(window):

    # Get the pitch contours and confidence curve from the contour
    pitch_contour, confidence_curve = compute_pitch_contour(window)

    # Make a histogram from the contours with the default number of bins
    hist = np.histogram(pitch_contour, bins=10)[0]

    # Compute the mean and standard deviation of pitch contours
    mean = np.mean(pitch_contour)
    std = np.std(pitch_contour)

    # Return the histogram followed by the mean and standard deviation
    return np.append(hist, [mean, std])

def compute_mfcc(window):
    mfccs = mfcc(window,8000,winstep=.0125)
    return mfccs

def compute_delta_coefficients(window, n=2):
    mfcc = compute_mfcc(window)
    numerator = [np.sum([(i * (mfcc[t + i,:] - mfcc[t - i, :])) for i in range(0, n + 1)], axis=0) for t in range(0 + n, len(mfcc) - n)]
    denominator = (2 * np.sum(np.array(range(1, n + 1)) ** 2))
    return numerator / denominator

for i,window_with_timestamp_and_label in enumerate(data):
    window = window_with_timestamp_and_label[1:-1] # get window without timestamp/label
    label = data[i,-1] # get label
    if(i == 0):
        print 'Formant Features'
        print np.shape(compute_formant_features(window))
        print 'Contour Features'
        print np.shape(compute_pitch_features(window))
        print 'MFCC'
        print np.shape(compute_mfcc(window))
        print 'Delta'
        print np.shape(compute_delta_coefficients(window))
