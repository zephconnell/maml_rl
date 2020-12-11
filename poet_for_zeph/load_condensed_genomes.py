import json
def load_condensed_genomes(file):
    D = []
    with open(file, 'r') as f:
        for l in f.readlines():
            D.append(json.loads(l))
    return D
