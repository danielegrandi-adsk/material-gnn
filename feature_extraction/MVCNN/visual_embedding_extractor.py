import os
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np


def get_all_files(directory, pattern):
    return [f for f in Path(directory).glob(pattern)]


def save_json(assembly_file, assembly_data, output_dir):
    output = Path(output_dir) / assembly_file.parts[-2] / 'assembly.json'
    os.makedirs(output.parent, exist_ok=True)
    with open(output, "w", encoding="utf8") as f:
        json.dump(assembly_data, f, indent=4)


def read_embeddings(paths):
    emb_files = []
    for path in paths:
        emb_files.extend(get_all_files(path, "*_embeddings.npy"))
    assert len(emb_files) > 0, "did not find embedding files"

    embeddings = {}
    for path in emb_files:
        data = np.load(str(path), allow_pickle=True)
        for row in data:
            embeddings[row[1]] = row[2]
    return embeddings


def map_bodies(jsons, process_all=False):
    errors = 0
    successes = 0
    for input_json in tqdm(jsons, desc='mapping bodies and saving output'):
        with open(input_json, "r", encoding="utf-8") as f:
            assembly_data = json.load(f)
        bodies = assembly_data['bodies'].keys()
        if not assembly_data['properties']['only_default_materials'] or process_all:
            for body in bodies:
                try:
                    embedding = embeddings[body].tolist()
                    successes += 1
                except KeyError:
                    embedding = None
                    errors += 1
                assembly_data['bodies'][body]["visual_embedding"] = embedding
            save_json(input_json, assembly_data, output_dir)
    return errors, successes


if __name__ == '__main__':
    input_dir = r"FusionGallery"
    output_dir = r"data"
    embeddings = read_embeddings([r"embeddings"])

    input_jsons = get_all_files(input_dir, "*/assembly.json")

    errors, successes = map_bodies(input_jsons)
    print(f"errors/total: {errors}/{errors+successes}")
