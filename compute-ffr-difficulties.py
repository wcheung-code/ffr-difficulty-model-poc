import os
import re
import numpy as np
import argparse
from collections import OrderedDict
from shutil import copy

receptors = {
    'left': 0,
    'down': 1,
    'up': 2,
    'right': 3
}

## Assumes all raw data (containing .mp3 and .sm files) provided in ./data/simfile_artists/{simfile_author}
## Raw data not provided in Git repository, all .sm files used for implementation provided in ./data/input/

def extract_data(simfile_authors, input_path, output_path):
    for simfile_author in simfile_authors:
        sm_directory = '{}/{}'.format(input_path, simfile_author)
        for root, dirs, files in os.walk(sm_directory):
            for file in files:
                if file.endswith(".sm"):
                    src = os.path.join(root, file)
                    copy(src, output_path)
    return True

def multiple_replace(dict, text):
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 

def initialize(data_points):
    results_dict = dict()
    mean_dict = {k: v.mean() for k, v in data_points.items()}
    for i, (k, v) in enumerate(sorted(mean_dict.items(), key=lambda a: a[1])):
        results_dict[k] = np.ceil((i + 0.001)*MAX_DIFFICULTY / len(data_points)).astype(int)
    return results_dict

def configure(source_dir, threshold, k, data_points = dict(), song_list = list()):
    for source in os.listdir(source_dir):
        absolute_file_path = "/".join([source_dir, source])
        timestamps = GetTimeStamps(absolute_file_path).timestamps
        point = np.empty(shape=(4, ), dtype=np.float)
        for i, orientation in enumerate(['left', 'down', 'up', 'right']):
            deltas = np.diff([v['timestamp'] for k, v in timestamps.items() if v['step'][receptors[orientation]] == '1'])
            deltas = deltas[np.nonzero(deltas)]
            point[i] = np.sort(deltas)[:threshold].mean()**(-k)
        point = np.nan_to_num(point, nan = point[np.logical_not(np.isnan(point))].mean())
        data_points[GetTimeStamps(absolute_file_path).title] = point
        song_list.append(GetTimeStamps(absolute_file_path).title)
    return data_points, song_list

class SMFileObject:
    def __init__(self, path):
        self.path = path
        self.bpm_dict = dict()
        self.step_dict = dict()

        with open (self.path, 'r') as sm_file:

            regex_dictionary = {
                "\n," : ",", "\n;" : ";"
            } 
            content = multiple_replace(regex_dictionary, sm_file.read())

        for line in content.split('\n'):
            if re.match(r'\/{2}-{15}', line): 
                break
            try:
                sm_dict_key, sm_dict_value = tuple(line.strip().split(':'))
                if not sm_dict_key == '#BPMS':
                    setattr(self, sm_dict_key.strip('#').lower(), sm_dict_value.strip(';'))                        
                else:
                    if ',' in sm_dict_value:
                        for bpm_change in sm_dict_value.split(','):
                            bpm_dict_key, bpm_dict_value = tuple(map(float, bpm_change.split('=')))
                            self.bpm_dict[bpm_dict_key] = bpm_dict_value
                    else:
                        bpm_dict_key, bpm_dict_value = tuple(map(float, sm_dict_value.strip(';').split('=')))
                        self.bpm_dict[bpm_dict_key] = bpm_dict_value
            except ValueError:
                continue

        metadata, step_dict = '', dict()
        with open(self.path, 'r') as sm_file:
            scribe = False
            for line in sm_file:
                if 'NOTES' in line:
                    scribe = True
                if scribe:
                    metadata += line
        
        raw_notes_metadata = metadata.split(':').pop()
        for i, notes in enumerate(raw_notes_metadata.split(',')):
            step_dict_value = re.sub(r'// measure [^\s]* ', r'', ' '.join(notes.strip().strip(';').split('\n')))
            self.step_dict[i] = step_dict_value.strip()

class GetTimeStamps(SMFileObject):
    
    def __init__(self, path):
        super().__init__(path)

        self.timestamps = dict()
        discretize = lambda ms: round(ms/1000 * 30)/30 * 1000
        
        self.filtered_step_dict = dict()
        existing_note, beat_number, delta = 0, 0, 0 
        for beat in self.step_dict.values():
            if '1' in beat:
                existing_note += 1
            if not existing_note and '1' not in beat:
                delta += 1
            if existing_note:
                self.filtered_step_dict[beat_number] = beat
                beat_number += 1
        
        self.bpm_dict = {(0.0 if k - delta < 0 else 4*float(k)): v for k, v in self.bpm_dict.items()}
        
        self.preprocessed_step_dict = OrderedDict()
        for j, beat in enumerate(self.filtered_step_dict.keys()):
            steps = self.filtered_step_dict[beat]
            quantization = len(steps.split(' '))
            for i, step in enumerate(steps.split(' ')):
                if step != '0000':
                    self.preprocessed_step_dict[4.0*(beat + float(i/quantization))] = step

        self.bpm_dict = OrderedDict({k: v for k, v in self.bpm_dict.items()
                                     if k < max(self.preprocessed_step_dict.keys())})
        
        total_ms = 0
        self.ms_timing = OrderedDict()
        bpm_beat_levels = list(self.bpm_dict.keys())
        beats = list(self.preprocessed_step_dict.keys())
        for i, bpm_beat in enumerate(bpm_beat_levels):
            if i + 1 == len(bpm_beat_levels):
                next_bpm_beat = max(self.preprocessed_step_dict.keys())
            else:
                next_bpm_beat = bpm_beat_levels[i + 1]
            bpm = self.bpm_dict[bpm_beat]
            ms_per_beat = 60000./bpm

            for beat in filter(lambda x: x >= bpm_beat and x < next_bpm_beat, beats):
                j = beats.index(beat) + 1
                if j + 1 == len(beats):
                    delta = discretize((beats[-1] - beat) * ms_per_beat)
                else:
                    delta = discretize((beats[j] - beat) * ms_per_beat)  
                self.ms_timing[beat] = total_ms
                total_ms += delta
        last_beat, _ = max(self.preprocessed_step_dict.items())
        if last_beat != max(self.ms_timing.keys()):
            self.ms_timing[last_beat] = discretize(total_ms + (last_beat - max(self.ms_timing.keys())) * ms_per_beat)
            
        for beat, step in self.preprocessed_step_dict.items():
            timestamp = self.ms_timing[beat]
            self.timestamps[beat] = {'timestamp': timestamp, 'step': step}
            
class EMAlgorithm:
    def __init__(self, source_dir, threshold = 100, k = 0.05):
        self.source_dir = source_dir
        self.threshold, self.k = threshold, k
        self.data_points, self.song_list = configure(source_dir, self.threshold, self.k)
        self.difficulties = initialize(self.data_points)
        
        self.centroids = dict()
        for difficulty in range(1, MAX_DIFFICULTY + 1):
            self.centroids[difficulty] = np.random.rand(4,)
            
    def get_parameters(self):
        return self.threshold, self.k
    
    def set_parameters(self, param):
        self.threshold = param['threshold']; self.k = param['k']
        self.difficulties = initialize(self.data_points)
        self.data_points, self.song_list = configure(self.source_dir, self.threshold, self.k)
        return True
    
    def get_difficulties(self):
        return self.difficulties
    
    def get_centroids(self):
        return self.centroids
    
    def expectation(self):
        for difficulty in range(1, MAX_DIFFICULTY + 1):
            sub_dictionary = {k: v for k, v in self.get_difficulties().items() if v == difficulty}
            centroid = np.mean([v for k, v in self.data_points.items() if k in sub_dictionary.keys()], axis = 0)
            self.centroids[difficulty] = centroid
            
    def maximization(self):
        delta = 0
        for k in self.get_difficulties().keys():
            mse = np.empty(shape=(MAX_DIFFICULTY, ), dtype=np.float)
            for difficulty in range(1, MAX_DIFFICULTY + 1):
                mse[difficulty - 1] = np.linalg.norm(self.data_points[k] - self.get_centroids()[difficulty])
            delta += np.abs(self.difficulties[k] - (np.argmin(mse) + 1))
            self.difficulties[k] = np.argmin(mse) + 1
        return delta
    
    def train(self, delta = 100):
        while delta > 0:
            self.expectation()
            delta = self.maximization()
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Assigning stepchart difficulties using EM Algorithm")
    parser.add_argument('-n', '--upperbound', type = int, metavar = '', help = "maximum assigned difficulty for all stepfiles")
    parser.add_argument('-s', '--source', type = str, metavar = '', help = "relative path of source files (in .sm format)")
    args = parser.parse_args()

    MAX_DIFFICULTY = args.upperbound
    sm_source_dir = args.source

###  NOTE: simfile_authors defined as list of usernames (commented out for code execution) 
#    input_directory = './data/simfile_artists'
#    simfile_authors = ['username1', 'username2', ...]
#    completed = extract_data(simfile_authors, input_directory, sm_source_dir)

    model = EMAlgorithm(sm_source_dir)

###  NOTE: provided added functionality to tune hyperparameters used in model
###  Code is currently using optimal hyperparameters
#    model.set_parameters({'threshold': 50, 'k': 0.01})

    model.train()
    results = model.difficulties
    
    print('Proposed stepfile difficulty ratings (max: {}): {}'.format(MAX_DIFFICULTY, results))
