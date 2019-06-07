import os
import json
import random
import numpy as np
from tqdm import tqdm
"""
Load data from json files. Return list of json loads from different files.
"""

BASE_DIR = 'data/'

FILES_TO_LOAD = []
for file in os.listdir(BASE_DIR):
    if not file[-7:] == '_0.json':
        continue
    FILES_TO_LOAD.append(os.path.join(BASE_DIR, file))

class Data:
    def __init__(self, small = False):
        """

        :param n: number of files to load
        """
        random.seed(10)
        self.small = small
        self.raw = None
        self.formatted = None
        self.statistics = None
        self._size = None
        self.train = None
        self.test = None
        self.train_size = None
        self.test_size = None
        self.ratio = 0.85 # train test ratio
        self.mode = 'train'
        self.load()
        self.format()
        self.split(ratio=self.ratio)
        self.stats()

    def load(self):
        """
        load a list of files into self.raw. Each loaded file is a d dictionary in the following format:
        {'video': path_of_video_analyzed, 'data': list_of_dictionaries_containing_info_about_detection}
        """
        self.raw = []
        if self.small:
            n = 3
        else:
            n = len(FILES_TO_LOAD)

        for i in tqdm(range(n)):
            next_file = FILES_TO_LOAD[i]
            data = []
            while True:
                with open(next_file, 'r') as f:
                    d = json.load(f)
                    data.extend(d['data'])
                    if 'cont_file' in d.keys() and os.path.isfile(d['cont_file']):
                        next_file = d['cont_file']
                    else:
                        break

            self.raw.append({'video': d['video'], 'data': data})

    def format(self):
        """
        Set self.formatted to a dictionary of episodes of tracking in the following format: {(video_path, track_id): list_of_encoding_objects}
        See Encoding class for encoding structure
        """
        self._size = 0
        key_set = set()
        self.formatted = {}
        for cur_data in self.raw:
            filename = cur_data['video']
            for d in cur_data['data']:
                if 'track_id' not in d.keys() or d['track_state'] != 2:
                    continue

                key = (filename, d['track_id'])

                if key not in key_set:
                    self.formatted[key] = []
                    key_set.add(key)
                encoding = Encoding(data = d)
                self.formatted[key].append(encoding)
                self._size += 1 # Increment size of dataset

    def stats(self):
        """
        Compute statistics that will be used by reward function in rewards.py. Store in self.statistics
        """
        from core import AbstractState
        confs = {'age': [], 'gender': []}
        changes_in_entropy = []
        for key, encodings in self.formatted.items():
            prev_state = None
            for enc in encodings:
                state = AbstractState(enc, prev_state=prev_state)
                state.update(np.random.randint(0, 2)) # Update state
                changes_in_entropy.append(state.entropy_change)
                confs['age'].append(np.max(enc.age_conf))
                confs['gender'].append(np.max(enc.gender_conf))

        self.statistics = (np.mean(confs['gender']),
                          np.mean(confs['age']),
                          np.mean(changes_in_entropy),
                          np.std(confs['gender']),
                          np.std(confs['age']),
                          np.std(changes_in_entropy))

    def split(self, ratio = 0.8):
        size = len(self.formatted)
        splits = [{}, {}]
        counts = [0, 0]
        cur_idx = 0
        change_flag = False
        for i, (key, value) in enumerate(self.formatted.items()):
            if not change_flag and i/size > ratio:
                cur_idx = 1
                change_flag = True
            splits[cur_idx][key] = value
            counts[cur_idx] += len(value)
        self.train = splits[0]
        self.test = splits[1]
        self.train_size = counts[0]
        self.test_size = counts[1]

    def __iter__(self):
        if self.mode == 'train':
            return iter(DataIterator(data = self.train))
        elif self.mode == 'test':
            return iter(DataIterator(data=self.test))

    def size(self):
        if self.mode == 'train':
            return self.train_size
        elif self.mode == 'test':
            return self.test_size
        else:
            return self._size


class Encoding:
    """
    Encoding data structure
    """
    def __init__(self, data = None):
        self.data = data
        self.feature = data['feature']
        self.box = data['box']
        self.gender = data['gender']
        self.age = data['age']
        self.gender_conf = data['gender_conf']
        self.age_conf = data['age_conf']
        self.frame_id = data['frame_id']



class DataIterator:
    """
    A class providing ease of iteration through formatted data, aka output of format_raw
    """
    def __init__(self, data = None):
        """
        :param data: Output from format_raw function defined in this file.
        """
        self.data = data
        self.keys_sequence = None
        self.idx = None

    class EpsiodeIterator:
        """
        Class iterates through episodes and returns tuples of encoding, done
        """
        def __init__(self, episode = None):
            self.episode = episode
            self.idx = None

        def __iter__(self):
            self.idx = 0
            return self

        def __next__(self):
            if self.idx >= self.__len__():
                raise StopIteration

            done = False
            if self.idx == self.__len__() - 1:
                done = True

            idx = self.idx
            self.idx += 1

            return self.episode[idx], done

        def __len__(self):
            return len(self.episode)

    def __iter__(self):
        self.keys_sequence = random.sample(list(self.data.keys()), k=self.__len__())
        self.idx = 0
        self.key = self.keys_sequence[self.idx]
        return self

    def __next__(self):
        if self.idx >= self.__len__():
            raise StopIteration
        idx = self.idx
        self.idx += 1
        self.key = self.keys_sequence[idx]
        return self.EpsiodeIterator(episode =  self.data[self.key]), self.key[0]

    def __len__(self):
        return len(self.data)

    def total_size(self):
        size = 0
        for key, epsiter in self.data.items():
            size += len(epsiter)
        return size





if __name__ == '__main__':
    # Test
    data = Data(n = 2)





