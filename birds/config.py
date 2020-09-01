import os
sample_rate = 32000
clip_length = 5  # clip samples into 5 sec data points
clip_samples = sample_rate * clip_length
window_size = 1024
hop_size = 320
mel_bins = 64
fmin = 50
fmax = 14000  # should we use low pass filter?
classes_num = 264

with open('/Users/shasha.lin/birds/names') as f:
    names = f.read().splitlines()

name2idx = {name: i for i, name in enumerate(names)}
idx2name = {i: name for i, name in enumerate(names)}

log_dir = os.path.abspath('./log')

total_audio_sample_train = 184804
