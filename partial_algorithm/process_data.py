import sys
import json
from pathlib import Path
from collections import Counter
import numpy as np
from tqdm import tqdm

from partial_algorithm.config import *

DOMINANT_MATERIALS = []

PREDICTION = None


def set_prediction(prediction):
    global PREDICTION
    PREDICTION = prediction


class AssemblyGraph:
    """
    Construct a graph representing an assembly with connectivity
    between B-Rep bodies with joints and contact surfaces
    """

    def __init__(self, assembly_data):

        if isinstance(assembly_data, dict):  # Check if assembly_data is a dictionary
            self.assembly_data = assembly_data
        else:
            if isinstance(assembly_data, str):
                assembly_file = Path(assembly_data)
            else:
                assembly_file = assembly_data
            assert assembly_file.exists()
            with open(assembly_file, "r", encoding="utf-8") as f:
                self.assembly_data = json.load(f)

        self.prediction = PREDICTION
        self.graph_nodes = []
        self.graph_links = []
        self.graph_edges = []
        self.graph_node_ids = set()
        self.depth = 0  # Depth of hierarchy (depth of recursion of "tree")
        self.train_test_split = self.assembly_data["train_test_split"]

    def get_graph_data(self):
        """Get the graph data as a list of nodes and links"""

        self.graph_nodes = []
        self.graph_links = []
        self.graph_edges = []
        self.graph_node_ids = set()

        self.populate_graph_nodes()  # Populate "nodes" list

        self.populate_graph_links()  # Populate "links" list

        # further processing of the graph links (edges)
        self.populate_graph_edges()

        # further processing of the graph links (edges)
        if SHARED_OCCURRENCE:
            self.populate_graph_shared_occ_links()

        if GLOBAL_FEATURES:
            self.populate_graph_global_features()

        return self.graph_nodes, self.graph_links, self.graph_edges, self.depth, self.train_test_split

    def get_graph_networkx(self):
        """Get a networkx graph"""
        graph_data = {
            "directed": True,
            "multigraph": False,
            "graph": {},
            "nodes": [],
            "links": [],
            "edges": [],
        }

        """Use graph edges (new implementation) or graph links (original)"""

        if not ORIGINAL_EDGE_FEATURES:
            graph_data["nodes"], _, graph_data["links"], _, _ = self.get_graph_data()  # processed edges as graph links
        else:
            graph_data["nodes"], graph_data["links"], _, _, _ = self.get_graph_data()  # original links as graph links
        from networkx.readwrite import json_graph
        return json_graph.node_link_graph(graph_data)

    def get_node_label_dict(self, attribute="occurrence_name"):
        """Get a dictionary mapping from node ids to a given attribute"""
        label_dict = {}
        if len(self.graph_nodes) == 0:
            return label_dict
        for node in self.graph_nodes:
            node_id = node["id"]
            if attribute in node:
                node_att = node[attribute]
            else:
                node_att = node["body_name"]
            label_dict[node_id] = node_att
        return label_dict

    def export_graph_json(self, json_file):
        """Export the graph as an networkx node-link format json file"""
        graph_data = {
            "directed": True,
            "multigraph": False,
            "graph": {},
            "nodes": [],
            "links": [],
            "edges": [],
        }
        graph_data["nodes"], graph_data["links"], graph_data["edges"], _, _ = self.get_graph_data()
        with open(json_file, "w", encoding="utf8") as f:
            json.dump(graph_data, f, indent=4)
        return json_file.exists()

    def get_graph_links(self):
        """Get the links"""

        graph_data = {
            "directed": True,
            "multigraph": False,
            "graph": {},
            "nodes": [],
            "links": [],
            "edges": [],
        }
        graph_data["nodes"], graph_data["links"], graph_data["edges"], _, _ = self.get_graph_data()
        return json.dumps(graph_data["links"])

    def populate_graph_nodes(self):
        """
        Recursively traverse the assembly tree
        and generate a flat set of graph nodes from bodies
        """
        # First get the root and add it's bodies

        root_component_uuid = self.assembly_data["root"]["component"]  # root > component

        root_component = self.assembly_data["components"][root_component_uuid]  # components > component uuid

        if "bodies" in root_component:  # Add each body of the root component as a node
            for body_uuid in root_component["bodies"]:
                node_data = self.get_graph_node_data(body_uuid)  # Get the node features
                self.graph_nodes.append(node_data)  # Append to the "nodes" list

        # Recurse over the occurrences in the tree
        tree_root = self.assembly_data["tree"]["root"]  # tree > root
        # Start with an identity matrix
        root_transform = np.identity(4)
        self.walk_tree(tree_root, root_transform)
        # Check all our node ids are unique
        total_nodes = len(self.graph_nodes)
        self.graph_node_ids = set([f["id"] for f in self.graph_nodes])  # set of unique ids in self.graph_nodes
        assert total_nodes == len(self.graph_node_ids), "Duplicate node ids found"  # all nodes are unique in ids

    def populate_graph_links(self):
        """Create links in the graph between bodies with joints and contacts"""
        if "joints" in self.assembly_data:
            self.populate_graph_joint_links()
        if "as_built_joints" in self.assembly_data:
            self.populate_graph_as_built_joint_links()
        if "contacts" in self.assembly_data:
            self.populate_graph_contact_links()

    def populate_graph_shared_occ_links(self):  # My own implementation
        for source in self.graph_nodes:
            for target in self.graph_nodes:
                if source == target:
                    continue
                if source["occ_id"] == target["occ_id"]:  # Shared occurrence
                    source_id = source["id"]
                    target_id = target["id"]
                    link_data = {"id": f"{source_id}>{target_id}", "source": source_id, "target": target_id,
                                 "type": "Shared Occurrence"}
                    self.graph_links.append(link_data)

    def populate_graph_global_features(self):  # My own implementation
        # For each node of this graph, add the graph's global features
        for node in self.graph_nodes:
            node["global_features"] = {}
            node["global_features"]["edge_count"] = self.assembly_data["properties"]["edge_count"]
            node["global_features"]["face_count"] = self.assembly_data["properties"]["face_count"]
            node["global_features"]["loop_count"] = self.assembly_data["properties"]["loop_count"]
            node["global_features"]["shell_count"] = self.assembly_data["properties"]["shell_count"]
            node["global_features"]["vertex_count"] = self.assembly_data["properties"]["vertex_count"]
            node["global_features"]["volume"] = self.assembly_data["properties"]["volume"]
            node["global_features"]["center_x"] = self.assembly_data["properties"]["center_of_mass"]["x"]
            node["global_features"]["center_y"] = self.assembly_data["properties"]["center_of_mass"]["y"]
            node["global_features"]["center_z"] = self.assembly_data["properties"]["center_of_mass"]["z"]
            node["global_features"]["products"] = self.assembly_data["properties"]["products"]
            node["global_features"]["categories"] = self.assembly_data["properties"]["categories"]
            node["global_features"]["industries"] = self.assembly_data["properties"]["industries"]
            node["global_features"]["likes_count"] = self.assembly_data["properties"]["likes_count"]
            node["global_features"]["comments_count"] = self.assembly_data["properties"]["comments_count"]
            node["global_features"]["views_count"] = self.assembly_data["properties"]["views_count"]

    def populate_graph_edges(self):  # My own implementation

        link_ids = set([f["id"] for f in self.graph_links])  # Remove duplicates

        for id in link_ids:

            edge_data = {}
            edge_data["id"] = id

            counter = 0
            for link in self.graph_links:
                if link["id"] == id:
                    counter += 1
                    edge_data["source"] = link["source"]
                    edge_data["target"] = link["target"]

                if "type" in link and "joint_type" in link:
                    edge_data["type"] = link["type"] + link["joint_type"]
                else:
                    if "joint_type" in link:
                        edge_data["type"] = link["joint_type"]
                    else:
                        edge_data["type"] = link["type"]

            edge_data["edge_weight"] = counter
            self.graph_edges.append(edge_data)

    def walk_tree(self, occ_tree, occ_transform):
        """Recursively walk the occurrence tree"""
        self.depth = 1

        for occ_uuid, occ_sub_tree in occ_tree.items():  # for each item from the tree root
            self.depth += 1
            occ = self.assembly_data["occurrences"][occ_uuid]  # occurrences > occurrence uuid
            if not occ["is_visible"]:  # not visible, ignore and move on
                continue

            # "@" means matrix multiplication
            occ_sub_transform = occ_transform @ self.transform_to_matrix(occ["transform"])

            if "bodies" in occ:  # Add each body in the occurrence as a graph node
                for occ_body_uuid, occ_body in occ["bodies"].items():
                    if not occ_body["is_visible"]:
                        continue
                    node_data = self.get_graph_node_data(
                        occ_body_uuid,
                        occ_uuid,
                        occ,
                        occ_sub_transform
                    )
                    self.graph_nodes.append(node_data)  # append node to "nodes" list
            self.walk_tree(occ_sub_tree, occ_sub_transform)  # recursion

    def get_graph_node_data(self, body_uuid, occ_uuid=None, occ=None, transform=None):
        """Add a body as a graph node"""
        # The node features considered:

        # id ({occ_uuid}_{body_uuid})
        # body_name (name)
        # body_file (body_uuid)
        # body_type (type)
        # occurrence_name (set by user, may need to use TechNet to extract keywords)
        # transform [ignored]

        # material (material-id)
        # material_name (material-name)
        # material_category (material-category)
        # appearance_id (appearance-id)
        # appearance_name (appearance-name)
        # obj (the name of the .obj file)
        # occurrence_area
        # occurrence_volume

        # valid_body_name
        # valid_occ_name

        # [Features added for a02.1 version dataset]
        # body_area
        # body_volume

        # [Features added for a02.2 and 3 version dataset: technet embeddings]
        # body_name_embedding
        # occ_name_embedding

        # [Features added for a02.8: visual embeddings]
        # visual_embedding

        # [Features added for a03.1: center of mass]

        body = self.assembly_data["bodies"][body_uuid]  # bodies > body uuid
        node_data = {}
        if occ_uuid is None:
            body_id = body_uuid
        else:
            body_id = f"{occ_uuid}_{body_uuid}"

        node_data["train_test"] = self.train_test_split
        node_data["occ_id"] = occ_uuid
        node_data["id"] = body_id
        node_data["body_name"] = body["name"]
        node_data["valid_body_name"] = str(body["valid_body_name"])
        node_data["material_category_tier1"] = body["material_category"]["tier1"]
        node_data["material_category"] = body["material_category"]
        node_data["body_type"] = body["type"]
        node_data["body_file"] = body_uuid

        try:
            node_data["body_area"] = body["physical_properties"]["area"]
            node_data["body_volume"] = body["physical_properties"]["volume"]
        except:
            print("[Warning]: Invalid body area/volume")
            node_data["body_area"] = 0
            node_data["body_volume"] = 0

        node_data["center_x"] = body["physical_properties"]["center_of_mass"]["x"]
        node_data["center_y"] = body["physical_properties"]["center_of_mass"]["y"]
        node_data["center_z"] = body["physical_properties"]["center_of_mass"]["z"]

        node_data["material_name"] = body["material"]["name"]
        node_data["appearance_id"] = body["appearance"]["id"]

        if TECHNET_EMBEDDING:
            try:
                node_data["body_name_embedding"] = body["body_name_embedding"]
            except:
                print("[Error]: No TechNet embedding detected!")
                exit(1)

        if VISUAL_EMBEDDING:

            try:
                if body["visual_embedding"] is None:
                    node_data["visual_embedding"] = [0] * 512
                else:
                    node_data["visual_embedding"] = body["visual_embedding"]
            except:
                print("[Error]: No visual embedding detected!")
                exit(1)

        # [Preprocessing] Modify appearance_id to correspond to the material_id pool

        try:
            node_data["appearance_id"] = node_data["appearance_id"].split('_')[0]
        except:
            node_data["appearance_id"] = node_data["appearance_id"]

        node_data["appearance_name"] = body["appearance"]["name"]
        node_data["obj"] = body["obj"]

        if self.prediction not in ["material_id", "material_category_tier_1", "material_category_full"]:
            print("Error: Invalid / No Prediction Selection")

        if self.prediction == "material_id":  # We want to predict the physical material ID

            node_data["material"] = body["material_label"]
            # [Preprocessing] Change default material to non-default appearance

            # if MATERIAL_TO_APPEARANCE:
            #     if node_data["material"] == "PrismMaterial-018":  # default material
            #         # non-default appearance
            #         if node_data["appearance_id"] != "PrismMaterial-018":
            #             node_data["material"] = node_data["appearance_id"]

            if GROUP_RARE_MATERIAL:
                if node_data["material"] not in DOMINANT_MATERIALS:
                    node_data["material"] = "Other"
                    node_data["material_name"] = "Other"

        if self.prediction == "material_category_tier_1":

            try:
                node_data["material"] = node_data["material_category"]["tier1"]
            except:
                node_data["material"] = "Unknown"

        if self.prediction == "material_category_full":
            full_category = node_data["material_category"]["tier1"] + node_data["material_category"]["tier2"] + \
                            node_data["material_category"]["tier3"]
            node_data["material"] = full_category

        if occ:
            node_data["occurrence_name"] = occ["name"]
            node_data["valid_occ_name"] = str(occ["valid_occ_name"])
            node_data["occurrence_area"] = occ["physical_properties"]["area"]
            node_data["occurrence_volume"] = occ["physical_properties"]["volume"]

            if TECHNET_EMBEDDING:
                try:
                    node_data["occ_name_embedding"] = occ["occ_name_embedding"]
                except:
                    print("No occurrence embedding detected!")
                    exit(1)

        else:
            node_data["occurrence_name"] = "Root"  # Body belongs to root occurrence
            node_data["valid_occ_name"] = "True"
            node_data["occurrence_area"] = 0
            node_data["occurrence_volume"] = 0

            if TECHNET_EMBEDDING:
                # TechNet embedding for ROOT
                node_data["occ_name_embedding"] = ROOT_TECHNET_EMBEDDING

        if transform is None:
            transform = np.identity(4)
        # we do not need "transform" matrix for now
        # node_data["transform"] = transform.tolist()

        return node_data

    def populate_graph_joint_links(self):
        """Populate directed links between bodies with joints"""
        if self.assembly_data["joints"] == None:
            pass
        else:
            for joint_uuid, joint in self.assembly_data["joints"].items():
                try:
                    ent1 = joint["geometry_or_origin_one"]["entity_one"]
                    ent2 = joint["geometry_or_origin_two"]["entity_one"]
                    # Don't add links when the bodies aren't visible
                    body1_visible = self.is_body_visible(ent1)
                    body2_visible = self.is_body_visible(ent2)
                    if not body1_visible or not body2_visible:
                        continue
                    link_data = self.get_graph_link_data(ent1, ent2)
                    link_data["type"] = "Joint"
                    link_data["joint_type"] = joint["joint_motion"]["joint_type"]
                    self.graph_links.append(link_data)
                except:
                    continue

    def populate_graph_as_built_joint_links(self):
        """Populate directed links between bodies with as built joints"""
        if self.assembly_data["as_built_joints"] == None:
            pass
        else:
            for joint_uuid, joint in self.assembly_data["as_built_joints"].items():
                geo_ent = None
                geo_ent_id = None
                # For non rigid joint types we will get geometry
                if "joint_geometry" in joint:
                    if "entity_one" in joint["joint_geometry"]:
                        geo_ent = joint["joint_geometry"]["entity_one"]
                        geo_ent_id = self.get_link_entity_id(geo_ent)

                occ1 = joint["occurrence_one"]
                occ2 = joint["occurrence_two"]
                body1 = None
                body2 = None
                if geo_ent is not None and "occurrence" in geo_ent:
                    if geo_ent["occurrence"] == occ1:
                        body1 = geo_ent["body"]
                    elif geo_ent["occurrence"] == occ2:
                        body2 = geo_ent["body"]

                # We only add links if there is a single body
                # in both occurrences
                if body1 is None:
                    body1 = self.get_occurrence_body_uuid(occ1)
                    if body1 is None:
                        continue
                if body2 is None:
                    body2 = self.get_occurrence_body_uuid(occ2)
                    if body2 is None:
                        continue
                # Don't add links when the bodies aren't visible
                body1_visible = self.is_body_visible(body_uuid=body1, occurrence_uuid=occ1)
                body2_visible = self.is_body_visible(body_uuid=body2, occurrence_uuid=occ2)
                if not body1_visible or not body2_visible:
                    continue
                ent1 = f"{occ1}_{body1}"
                ent2 = f"{occ2}_{body2}"
                link_id = f"{ent1}>{ent2}"
                link_data = {}
                link_data["id"] = link_id
                link_data["source"] = ent1
                assert link_data["source"] in self.graph_node_ids, "Link source id doesn't exist in nodes"
                link_data["target"] = ent2
                assert link_data["target"] in self.graph_node_ids, "Link target id doesn't exist in nodes"
                link_data["type"] = "AsBuiltJoint"
                link_data["joint_type"] = joint["joint_motion"]["joint_type"]
                self.graph_links.append(link_data)

    def populate_graph_contact_links(self):
        """Populate undirected links between bodies in contact"""
        if self.assembly_data["contacts"] == None:
            pass
        else:
            for contact in self.assembly_data["contacts"]:
                ent1 = contact["entity_one"]
                ent2 = contact["entity_two"]
                # Don't add links when the bodies aren't visible
                body1_visible = self.is_body_visible(ent1)
                body2_visible = self.is_body_visible(ent2)
                if not body1_visible or not body2_visible:
                    continue
                link_data = self.get_graph_link_data(ent1, ent2)
                link_data["type"] = "Contact"
                self.graph_links.append(link_data)
                # Add a link in reverse so we have a undirected connection
                link_data = self.get_graph_link_data(ent2, ent1)
                link_data["type"] = "Contact"
                self.graph_links.append(link_data)

    def get_graph_link_data(self, entity_one, entity_two):
        """Get the common data for a graph link from a joint or contact"""
        link_data = {}
        link_data["id"] = self.get_link_id(entity_one, entity_two)
        link_data["source"] = self.get_link_entity_id(entity_one)
        assert link_data["source"] in self.graph_node_ids, "Link source id doesn't exist in nodes"
        link_data["target"] = self.get_link_entity_id(entity_two)
        assert link_data["target"] in self.graph_node_ids, "Link target id doesn't exist in nodes"
        return link_data

    def get_link_id(self, entity_one, entity_two):
        """Get a unique id for a link"""
        ent1_id = self.get_link_entity_id(entity_one)
        ent2_id = self.get_link_entity_id(entity_two)
        return f"{ent1_id}>{ent2_id}"

    def get_link_entity_id(self, entity):
        """Get a unique id for one side of a link"""
        if "occurrence" in entity:
            return f"{entity['occurrence']}_{entity['body']}"
        else:
            return entity["body"]

    def get_occurrence_body_uuid(self, occurrence_uuid):
        """Get the body uuid from an occurrence"""
        occ = self.assembly_data["occurrences"][occurrence_uuid]
        # We only return a body_uuid if there is one body
        if "bodies" not in occ:
            return None
        if len(occ["bodies"]) != 1:
            return None
        # Return the first key
        return next(iter(occ["bodies"]))

    def is_body_visible(self, entity=None, body_uuid=None, occurrence_uuid=None):
        """Check if a body is visible"""
        if body_uuid is None:
            body_uuid = entity["body"]
        if occurrence_uuid is None:
            # If we don't have an occurrence
            # we need to look in the root component
            if "occurrence" not in entity:
                body = self.assembly_data["root"]["bodies"][body_uuid]
                return body["is_visible"]
            # First check the occurrence is visible
            occurrence_uuid = entity["occurrence"]
        occ = self.assembly_data["occurrences"][occurrence_uuid]
        if not occ["is_visible"]:
            return False
        body = occ["bodies"][body_uuid]
        return body["is_visible"]

    def transform_to_matrix(self, transform=None):
        """
        Convert a transform dict into a
        4x4 affine transformation matrix
        """
        if transform is None:
            return np.identity(4)
        x_axis = self.transform_vector_to_np(transform["x_axis"])
        y_axis = self.transform_vector_to_np(transform["y_axis"])
        z_axis = self.transform_vector_to_np(transform["z_axis"])
        translation = self.transform_vector_to_np(transform["origin"])
        translation[3] = 1.0
        return np.transpose(np.stack([x_axis, y_axis, z_axis, translation]))

    def transform_vector_to_np(self, vector):
        x = vector["x"]
        y = vector["y"]
        z = vector["z"]
        h = 0.0
        return np.array([x, y, z, h])


def get_input_files(input="data"):
    """Get the input files to process, and populate global DOMINANT_MATERIALS"""
    global DOMINANT_MATERIALS
    input_path = Path(input)

    all_material_ids = []

    if not input_path.exists():
        sys.exit("Input folder/file does not exist")
    if input_path.is_dir():

        # Acquire the JSON file
        assembly_files = [f for f in input_path.glob("**/assembly.json")]

        if len(assembly_files) == 0:
            sys.exit("Input folder/file does not contain assembly.json files")

        # Find the TOP dominant materials (NOT rewriting the files)
        for assembly in tqdm(assembly_files, desc="Getting Input Files"):
            with open(assembly, "r", encoding="utf-8") as f:
                assembly_data = json.load(f)

                # Deal with NaN values (replace with "null")
                # assembly_string = json.dumps(assembly_data)
                # assembly_string = assembly_string.replace("NaN", '"null"')
                # assembly_data = json.loads(assembly_string)
                for body_uuid, body in assembly_data["bodies"].items():
                    all_material_ids.append(body["material_label"])

            # if not MATERIAL_TO_APPEARANCE:
            #     for body_uuid, body in assembly_data["bodies"].items():
            #         all_material_ids.append(body["material"]["id"])
            # else:
            #     for body_uuid, body in assembly_data["bodies"].items():
            #
            #         # Change appearance ID to better match the material ID pool:
            #
            #         try:
            #             body["appearance"]["id"] = body["appearance"]["id"].split('_')[0]
            #         except:
            #             body["appearance"]["id"] = body["appearance"]["id"]
            #
            #         # Change default material to non-default appearance
            #         if body["material"]["id"] == "PrismMaterial-018":  # default material
            #             # non-default appearance
            #             if body["appearance"]["id"] != "PrismMaterial-018":
            #                 body["material"]["id"] = body["appearance"]["id"]
            #
            #         all_material_ids.append(body["material"]["id"])

        dominant_materials = [material_id for material_id, material_id_count in
                              Counter(all_material_ids).most_common(DOMINANT_NUM)]
        DOMINANT_MATERIALS = dominant_materials

        return assembly_files

    elif input_path.name == "assembly.json":
        return [input_path]
    else:
        sys.exit("Input folder/file invalid")


def assembly2graph(path=DATA_PATH):
    """Convert assemblies (assembly.json) to graph format"""
    """Return a list of NetworkX graphs"""

    graphs = []

    input_files = get_input_files(path)

    for input_file in tqdm(input_files, desc="Generating Graphs"):
        ag = AssemblyGraph(input_file)

        graph = ag.get_graph_networkx()  # Change to corresponding NetworkX graphs

        graphs.append(graph)

    return graphs, input_files
