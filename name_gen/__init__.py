import json
import numpy as np

import os
import name_gen

file_dir = os.path.dirname(name_gen.__file__)

animals = json.load(open(file_dir+'/animals.json', 'r'))
adjectives = json.load(open(file_dir+'/adjectives.json', 'r'))


def get_name():
    adjective = np.random.choice(adjectives, 1)[0]
    animal = np.random.choice(animals, 1)[0]
    return '{}_{}'.format(adjective, animal)