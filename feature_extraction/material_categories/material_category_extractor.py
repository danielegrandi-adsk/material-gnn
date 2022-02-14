import os
from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd


def get_all_files(directory, pattern):
    return [f for f in Path(directory).glob(pattern)]


def save_json(assembly_file, assembly_data):
    output = Path(output_dir) / assembly_file.parts[-2] / 'assembly.json'
    os.makedirs(output.parent, exist_ok=True)
    with open(output, "w", encoding="utf8") as f:
        json.dump(assembly_data, f, indent=4)


def read_library(data_type):
    with open("materialData.json", "r", encoding="utf-8") as f:
        material_data = json.load(f)

    library_materials = {}
    for library in material_data:
        for material in library["materials"]:
            library_materials[material[data_type]] = material['properties'][11]['value']
    return library_materials


def read_appearance_library():
    return dict(pd.read_csv("appearanceData.csv").values)


def map_bodies(jsons, library):
    counter_material = []
    counter_mfg_method = []
    counter_material_label = []
    counter_all_default_materials = 0
    for input_json in tqdm(jsons, desc='mapping bodies and saving output'):
        with open(input_json, "r", encoding="utf-8") as f:
            assembly_data = json.load(f)
        bodies = assembly_data['bodies'].keys()
        default_material_count = 0
        for body in bodies:
            appearance_id = assembly_data['bodies'][body]["appearance"]["id"]
            material_id = assembly_data['bodies'][body]["material"]["id"]
            material_name = assembly_data['bodies'][body]["material"]["name"]
            material_appearance_id = assembly_data['bodies'][body]["material"]["appearance"]["id"]


            material_label = None
            if material_id != "PrismMaterial-018" and material_appearance_id != "PrismMaterial-018_physmat_aspects:Prism-256":
                try:
                    material_category = library[material_id]
                    material_label = material_id
                except KeyError:
                    try:
                        material_category = library[material_id.split("_")[0]]
                        material_label = material_id.split("_")[0]
                    except KeyError:
                        try:
                            material_category = library[material_name]
                            material_label = material_id
                        except KeyError:
                            try:
                                material_category = library[material_appearance_id]
                                material_label = material_appearance_id
                            except KeyError:
                                try:
                                    material_category = library[material_appearance_id.split("_")[0]]
                                    material_label = material_appearance_id.split("_")[0]
                                except KeyError:
                                    try:
                                        material_category = library[appearance_id]
                                        material_label = appearance_id
                                    except KeyError:
                                        try:
                                            material_category = library[appearance_id.split("_")[0]]
                                            material_label = appearance_id.split("_")[0]
                                        except KeyError:
                                            material_category = 'Unknown'
                                            material_label = material_id
            else:   # The body material is default, so we prioritize appearance over material
                # if library == library_material:
                #     default_material_count += 1
                try:
                    material_category = library[appearance_id]
                    material_label = appearance_id
                except KeyError:
                    try:
                        material_category = library[appearance_id.split("_")[0]]
                        material_label = appearance_id.split("_")[0]
                    except KeyError:
                        try:
                            material_category = library[material_appearance_id]
                            material_label = material_appearance_id
                        except KeyError:
                            try:
                                material_category = library[material_appearance_id.split("_")[0]]
                                material_label = material_appearance_id.split("_")[0]
                            except KeyError:
                                try:
                                    material_category = library[material_id]
                                    material_label = material_id
                                except KeyError:
                                    try:
                                        material_category = library[material_id.split("_")[0]]
                                        material_label = material_id.split("_")[0]
                                    except KeyError:
                                        try:
                                            material_category = library[material_name]
                                            material_label = material_id
                                        except KeyError:
                                            material_category = 'Unknown'
                                            material_label = material_id

                if library == library_material and (material_label == 'PrismMaterial-018' or
                                                    material_label == "PrismMaterial-018_physmat_aspects:Prism-256"):
                    default_material_count += 1
                if material_category == "":
                    material_category = 'Unknown'
                if library == library_material:
                    tier1, tier2, tier3 = 'Unknown', 'Unknown', 'Unknown'
                    tier1 = material_category.split(".")[0]
                    try:
                        tier2 = material_category.split(".")[1]
                    except IndexError:
                        tier2 = 'Unknown'
                    try:
                        tier3 = material_category.split(".")[2]
                    except IndexError:
                        tier3 = 'Unknown'
                    assembly_data['bodies'][body]["material_category"] = {"tier1": tier1, "tier2": tier2, "tier3": tier3}
                    counter_material.append(material_category)
                    assembly_data['bodies'][body]["material_label"] = material_label
                    counter_material_label.append(material_label)
                else:
                    assembly_data['bodies'][body]["manufacturing_method"] = material_category
                    counter_mfg_method.append(material_category)

        if default_material_count == len(bodies):
            assembly_data['properties']["only_default_materials"] = True
            counter_all_default_materials += 1
        else:
            assembly_data['properties']["only_default_materials"] = False

        save_json(input_json, assembly_data)
    return counter_material, counter_mfg_method, counter_material_label, counter_all_default_materials


def read_autodesk_library():
    materials_library = read_library("id")
    material_name_library = read_library("name")
    appearance_library = read_appearance_library()

    library = {}
    for lib in [materials_library, appearance_library, material_name_library]:
        for k, v in list(lib.items()):
            v = v.replace("/", ".").replace("Autodesk.Material Classifications.", "").replace(" ", "_")
            library[k] = v

    return library


if __name__ == '__main__':
    input_dir = r"FusionGallery"
    output_dir = r"data"

    library_material = read_autodesk_library()

    input_jsons = get_all_files(input_dir, "*/assembly.json")

    # map materials
    counter_material, counter_mfg_method, counter_material_label, counter_all_default_materials = \
        map_bodies(input_jsons, library_material)

    print(f"Assemblies with all default materials: {counter_all_default_materials}/{len(input_jsons)}")
    print(f"Useful assemblies: {len(input_jsons) - counter_all_default_materials}")






