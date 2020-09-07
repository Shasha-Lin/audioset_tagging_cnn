from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import time
from collections import defaultdict
import pickle
from functools import reduce
import h5py
import librosa
import logging
from tqdm import tqdm
from birds import config
from collections import OrderedDict
from utils.utilities import create_logging

TRAIN_AUDIO_PATH = '~/birds/train_audio'
train_size = .8
train_files = 'train.json'
val_files = 'val.json'
seed = 2020

np.random.seed(seed)

hdf5_path = os.path.expanduser('~/audioset_tagging_cnn/birds/full')


def output_hdf5(json_data_path, hdf5_path=hdf5_path, audio_size_file=None,
                # file storing the number of audio samples of clip_length in each class
                train=True, clip_length=config.clip_length, mini_data=0):
    log_dir = os.sep.join((config.log_dir, 'output_hdf5'))
    create_logging(log_dir, "w")
    if train:
        hdf5_path = os.sep.join((hdf5_path, 'train'))
    else:
        hdf5_path = os.sep.join((hdf5_path, 'val'))
    logging.info(f'writing hdf5 files to {hdf5_path}')
    file_dict = pickle.load(open(json_data_path, "rb"))
    audio_samples_num = 0
    if audio_size_file:
        with open(audio_size_file, 'rb') as f:
            audio_size_dict = pickle.load(f)
        for name, size in audio_size_dict.items():
            class_size = mini_data if mini_data else size
            audio_samples_num += class_size
    else:
        class_size_file = os.sep.join((os.path.dirname(hdf5_path), 'class_size_train' if train else 'class_size_val'))
        logging.info(f'will write class audio number to {class_size_file}')
        audio_size_dict = dict()
        if mini_data:
            audio_samples_num = mini_data * config.classes_num
        else:
            for name, file_list in file_dict.items():
                for file in file_list:
                    try:
                        class_size = int(librosa.core.get_duration(filename=file) / clip_length)
                        audio_size_dict[name] = class_size
                        audio_samples_num += class_size
                    except KeyboardInterrupt:
                        exit(1)
                    except:
                        logging.warning(f'Could not read {file}')
                logging.info(f'Total audio samples number after processing {name}: {audio_samples_num}')
        with open(class_size_file, 'wb') as f:
            pickle.dump(audio_size_dict, f)
    start_time = time.time()
    with h5py.File(hdf5_path, "w") as hf:
        hf.create_dataset('audio_name', shape=((audio_samples_num,)), dtype='S20')
        hf.create_dataset("waveform", shape=((audio_samples_num, config.clip_samples)), dtype=np.int16)
        hf.create_dataset('target', shape=((audio_samples_num, config.classes_num)), dtype=np.bool)
        hf.attrs.create('sample_rate', data=config.sample_rate, dtype=np.int32)

        audio_idx = 0
        files = reduce(lambda list1, list2: list1 + list2, file_dict.values())
        if mini_data:
            class_counter = defaultdict(int)
        for file in tqdm(files):
            if mini_data:
                if class_counter.keys() == audio_size_dict.keys() and sum(class_counter.values()) == audio_samples_num:
                    break
            try:
                audio, _ = librosa.core.load(file, sr=config.sample_rate, mono=True)
            except KeyboardInterrupt:
                exit(0)
            except:
                logging.warning(f'Cannot load file {file}')
            else:
                filename = file.split(os.sep)[-1]
                name = file.split(os.sep)[-2]
                for i in range(int(len(audio) / config.clip_samples)):
                    if mini_data and class_counter[name] >= mini_data:
                        break
                    audio_name = '--'.join((name, filename)).encode()
                    audio_clip = audio[i * config.clip_samples: (i + 1) * config.clip_samples]
                    target = np.zeros(config.classes_num, dtype=np.bool)
                    target[config.name2idx[name]] = 1

                    hf['audio_name'][audio_idx] = audio_name
                    hf['waveform'][audio_idx] = audio_clip
                    hf['target'][audio_idx] = target
                    i += 1
                    audio_idx += 1
                    if mini_data:
                        class_counter[name] += 1
                        break

    logging.info('Write to {}'.format(hdf5_path))
    logging.info('Pack hdf5 time: {:.3f}'.format(time.time() - start_time))


def train_val_split():
    train_audio_path = os.path.expanduser(TRAIN_AUDIO_PATH)
    names = os.listdir(train_audio_path)
    train = defaultdict(list)
    val = defaultdict(list)
    classes_num = 0
    for name in names:
        name_path = os.path.join(train_audio_path, name)
        if not os.path.isdir(name_path):
            continue
        classes_num += 1
        files = os.listdir(name_path)
        for file in files:
            file_path = os.path.join(name_path, file)
            if np.random.random() < train_size:
                train[name].append(file_path)
            else:
                val[name].append(file_path)
    try:
        assert len(train) == len(val) == classes_num
    except AssertionError:
        return train_val_split()
    with open(train_files, 'wb') as f:
        pickle.dump(train, f)

    with open(val_files, 'wb') as f:
        pickle.dump(val, f)
    return train, val, classes_num


# This is not used
class BirdsDataset(Dataset):

    def __init__(self, data_dict):
        """data_dict = {name: [list of files]}"""
        data_dict = OrderedDict(data_dict)
        counts = {name: len(files) for name, files in data_dict.items()}
        total = sum(counts)
        weights = {name: count/total for name, count in counts.items()}


