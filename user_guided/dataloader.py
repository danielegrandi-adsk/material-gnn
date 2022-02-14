from user_guided.process_data import *
import torch.nn as nn
import torch
import networkx as nx
from collections import Counter
from torch_geometric.data import DataLoader, Data
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tqdm import tqdm
from user_guided.config import ABLATED_FEATURES, GLOBAL_FEATURES
from random import sample
import pickle
import torch.nn.functional
import os
import random

LAZY_LOADING = False

"""Reproducibility"""
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def set_experiment(experiment):
    global ABLATED_FEATURES
    ABLATED_FEATURES = experiment


def set_material_tier(tier):
    global MATERIAL_TIER
    MATERIAL_TIER = tier


def get_vocab():
    """Get the vocab dictionary, weights dictionary, and list of NetworkX graphs"""

    graphs, input_files = assembly2graph()
    material_count = []

    vocab = {

        # Node features
        'material': set(),
        'appearance_id': set(),
        'appearance_name': set(),
        'body_name': set(),
        'body_type': set(),
        'occurrence_name': set(),
        'material_category_tier1': set(),
        'material_category_tier2': set(),
        'material_category_tier3': set(),

        # Edge features
        'edge_type': set(),

        # Global features
        'products': set(),
        'categories': set(),
        'industries': set(),
    }

    for input_file in tqdm(input_files, desc="Preprocessing Graphs"):

        ag = AssemblyGraph(input_file)
        nodes, edges, depth, train_test = ag.get_graph_data()

        for node in nodes:
            vocab['material'].add(node['material'])
            vocab['appearance_id'].add(node['appearance_id'])
            vocab['appearance_name'].add(node['appearance_name'])
            vocab['material_category_tier1'].add(node['material_category']['tier1'])
            vocab['material_category_tier2'].add(node['material_category']['tier2'])
            vocab['material_category_tier3'].add(node['material_category']['tier3'])
            vocab['body_name'].add(node['body_name'])
            vocab['body_type'].add(node['body_type'])
            vocab['products'].update(node["global_features"]["products"])
            vocab['categories'].update(node["global_features"]["categories"])
            vocab['industries'].update(node["global_features"]["industries"])

            try:
                vocab['occurrence_name'].add(node['occurrence_name'])
            except:
                vocab['occurrence_name'].add('')

            material_count.append(node['material'])

        for edge in edges:
            if 'type' in edge:
                vocab['edge_type'].add(edge['type'])

    for k, v in vocab.items():
        vocab[k] = {s: idx for idx, s in enumerate(sorted(v))}

    material_count = [vocab['material'][l] for l in material_count]
    material_w = [0.] * len(vocab['material'])

    for k, v in Counter(material_count).items():
        material_w[k] = len(material_count) / v

    weights = {0: torch.tensor(material_w)}

    return graphs, vocab, weights


def preprocess_data():
    processed_graphs = []
    data, vocab, weights = get_vocab()
    graph_num, node_num, edge_num = 0, 0, 0

    for graph in tqdm(data, desc="Encoding Features"):

        if graph.number_of_nodes() < 3 or graph.number_of_edges() < 2:
            continue

        train_test = None
        for node in graph.nodes(data=True):
            train_test = node[-1]["train_test"]
            break

        nodes, edges = [], []

        graph_num += 1
        node_num += graph.number_of_nodes()
        edge_num += graph.number_of_edges()

        mappings = {n: idx for idx, n in enumerate(graph.nodes())}
        graph = nx.relabel_nodes(graph, mappings)

        """Global features - if applicable"""

        if GLOBAL_FEATURES:
            global_edge_count_tensor = torch.tensor(
                [graph.nodes(data=True)[1]['global_features']["edge_count"]], dtype=torch.float)
            global_face_count_tensor = torch.tensor(
                [graph.nodes(data=True)[1]['global_features']["face_count"]], dtype=torch.float)
            global_loop_count_tensor = torch.tensor(
                [graph.nodes(data=True)[1]['global_features']["loop_count"]], dtype=torch.float)
            global_shell_count_tensor = torch.tensor(
                [graph.nodes(data=True)[1]['global_features']["shell_count"]], dtype=torch.float)
            global_vertex_count_tensor = torch.tensor(
                [graph.nodes(data=True)[1]['global_features']["vertex_count"]], dtype=torch.float)
            global_volume_tensor = torch.tensor(
                [graph.nodes(data=True)[1]['global_features']["volume"]], dtype=torch.float)
            global_center_x_tensor = torch.tensor(
                [graph.nodes(data=True)[1]['global_features']["center_x"]], dtype=torch.float)
            global_center_y_tensor = torch.tensor(
                [graph.nodes(data=True)[1]['global_features']["center_y"]], dtype=torch.float)
            global_center_z_tensor = torch.tensor(
                [graph.nodes(data=True)[1]['global_features']["center_z"]], dtype=torch.float)
            products = graph.nodes(data=True)[1]["global_features"]["products"]
            global_products = nn.functional.one_hot(torch.tensor([vocab['products'][prod] for prod in products]),
                                                    num_classes=len(vocab['products'])).sum(dim=0).float()
            categories = graph.nodes(data=True)[1]["global_features"]["categories"]
            global_categories = nn.functional.one_hot(torch.tensor([vocab['categories'][cat] for cat in categories]),
                                                      num_classes=len(vocab['categories'])).sum(dim=0).float()
            industries = graph.nodes(data=True)[1]["global_features"]["industries"]
            if len(industries) > 0:
                global_industries = nn.functional.one_hot(
                    torch.tensor([vocab['industries'][ind] for ind in industries]),
                    num_classes=len(vocab['industries'])).sum(dim=0).float()
            else:
                global_industries = torch.tensor(len(vocab['industries']) * [0]).long()
            global_likes_count = torch.tensor([graph.nodes(data=True)[1]['global_features']["likes_count"]],
                                              dtype=torch.float)
            global_comments_count = torch.tensor([graph.nodes(data=True)[1]['global_features']["comments_count"]],
                                                 dtype=torch.float)
            global_views_count = torch.tensor([graph.nodes(data=True)[1]['global_features']["views_count"]],
                                              dtype=torch.float)

            global_features_tensor = torch.cat((
                global_edge_count_tensor,
                global_face_count_tensor,
                global_loop_count_tensor,
                global_shell_count_tensor,
                global_vertex_count_tensor,
                global_volume_tensor,
                global_center_x_tensor,
                global_center_y_tensor,
                global_center_z_tensor,
                global_products,
                global_categories,
                global_industries,
                global_likes_count,
                global_comments_count,
                global_views_count
            ), -1).reshape(-1, 1)

            global_features_scaler = preprocessing.StandardScaler().fit(global_features_tensor)
            global_features_scaled = torch.from_numpy(global_features_scaler.transform(global_features_tensor))
            global_features_scaled = global_features_scaled.transpose(0, 1).repeat(len(graph.nodes), 1)

        """FLOAT features: occurrence_area"""
        occurrence_area_tensor = torch.tensor([[n[-1]['occurrence_area']] for n in graph.nodes(data=True)],
                                              dtype=torch.float)

        occurrence_area_scaler = preprocessing.StandardScaler().fit(occurrence_area_tensor)
        occurrence_area_scaled = occurrence_area_scaler.transform(occurrence_area_tensor)
        occurrence_area_scaled = torch.from_numpy(occurrence_area_scaled)

        """FLOAT features: occurrence_volume"""
        occurrence_volume_tensor = torch.tensor([[n[-1]['occurrence_volume']] for n in graph.nodes(data=True)],
                                                dtype=torch.float)

        occurrence_volume_scaler = preprocessing.StandardScaler().fit(occurrence_volume_tensor)
        occurrence_volume_scaled = occurrence_volume_scaler.transform(occurrence_volume_tensor)
        occurrence_volume_scaled = torch.from_numpy(occurrence_volume_scaled)

        """Including TechNet embeddings"""
        if TECHNET_EMBEDDING:
            body_name_embeddings = torch.tensor([[n[-1]['body_name_embedding']] for n in graph.nodes(data=True)],
                                                dtype=torch.float)

            occ_name_embeddings = torch.tensor([[n[-1]['occ_name_embedding']] for n in graph.nodes(data=True)],
                                               dtype=torch.float)

            body_name_embeddings = torch.squeeze(body_name_embeddings)
            occ_name_embeddings = torch.squeeze(occ_name_embeddings)

            body_name_embeddings_scalar = preprocessing.StandardScaler().fit(body_name_embeddings)
            body_name_embeddings_scaled = body_name_embeddings_scalar.transform(body_name_embeddings)
            body_name_embeddings_scaled = torch.from_numpy(body_name_embeddings_scaled)

            occ_name_embeddings_scalar = preprocessing.StandardScaler().fit(occ_name_embeddings)
            occ_name_embeddings_scaled = occ_name_embeddings_scalar.transform(occ_name_embeddings)
            occ_name_embeddings_scaled = torch.from_numpy(occ_name_embeddings_scaled)

        """FLOAT features: center_of_mass"""
        center_x = torch.tensor([[n[-1]['center_x']] for n in graph.nodes(data=True)], dtype=torch.float)
        center_y = torch.tensor([[n[-1]['center_y']] for n in graph.nodes(data=True)], dtype=torch.float)
        center_z = torch.tensor([[n[-1]['center_z']] for n in graph.nodes(data=True)], dtype=torch.float)

        center_x_scaler = preprocessing.StandardScaler().fit(center_x)
        center_y_scaler = preprocessing.StandardScaler().fit(center_y)
        center_z_scaler = preprocessing.StandardScaler().fit(center_z)

        center_x_scaled = torch.from_numpy(center_x_scaler.transform(center_x))
        center_y_scaled = torch.from_numpy(center_y_scaler.transform(center_y))
        center_z_scaled = torch.from_numpy(center_z_scaler.transform(center_z))

        """FLOAT features: body_area"""
        body_area_tensor = torch.tensor([[n[-1]['body_area']] for n in graph.nodes(data=True)], dtype=torch.float)

        body_area_scaler = preprocessing.StandardScaler().fit(body_area_tensor)
        body_area_scaled = body_area_scaler.transform(body_area_tensor)
        body_area_scaled = torch.from_numpy(body_area_scaled)

        """FLOAT features: body_volume"""
        body_volume_tensor = torch.tensor([[n[-1]['body_volume']] for n in graph.nodes(data=True)],
                                          dtype=torch.float)

        body_volume_scaler = preprocessing.StandardScaler().fit(body_volume_tensor)
        body_volume_scaled = body_volume_scaler.transform(body_volume_tensor)
        body_volume_scaled = torch.from_numpy(body_volume_scaled)

        """Including visual embeddings"""
        if VISUAL_EMBEDDING:
            try:
                visual_embeddings = torch.tensor([[n[-1]['visual_embedding']] for n in graph.nodes(data=True)],
                                                 dtype=torch.float)
            except:
                print("Warning: error when encoding visual embedding")
                visual_embedding_none = [[0 for i in range(512)]]
                visual_embeddings = [[n[-1]['visual_embedding']] for n in graph.nodes(data=True)]
                visual_embeddings = torch.tensor(
                    [visual_embedding_none if emb[0] is None else emb for emb in visual_embeddings], dtype=torch.float)

            visual_embeddings = torch.squeeze(visual_embeddings)

            visual_embeddings_scalar = preprocessing.StandardScaler().fit(visual_embeddings)
            visual_embeddings_scaled = visual_embeddings_scalar.transform(visual_embeddings)
            visual_embeddings_scaled = torch.from_numpy(visual_embeddings_scaled)

        material_category_tier1 = nn.functional.one_hot(torch.tensor(
            [vocab['material_category_tier1'][n[-1]['material_category_tier1']] for n in graph.nodes(data=True)]),
            len(vocab['material_category_tier1']))

        material_category_tier2 = nn.functional.one_hot(torch.tensor(
            [vocab['material_category_tier2'][n[-1]['material_category_tier2']] for n in graph.nodes(data=True)]),
            len(vocab['material_category_tier2']))

        material_category_tier3 = nn.functional.one_hot(torch.tensor(
            [vocab['material_category_tier3'][n[-1]['material_category_tier3']] for n in graph.nodes(data=True)]),
            len(vocab['material_category_tier3']))

        features_to_include = []

        if GLOBAL_FEATURES and "global_features" not in ABLATED_FEATURES:
            features_to_include.append(global_features_scaled)

        if "body_name" not in ABLATED_FEATURES:
            features_to_include.append(body_name_embeddings_scaled)

        if "occ_name" not in ABLATED_FEATURES:
            features_to_include.append(occ_name_embeddings_scaled)

        if "center_of_mass" not in ABLATED_FEATURES:
            features_to_include.append(center_x_scaled)
            features_to_include.append(center_y_scaled)
            features_to_include.append(center_z_scaled)

        if "body_physical_properties" not in ABLATED_FEATURES:
            features_to_include.append(body_area_scaled)
            features_to_include.append(body_volume_scaled)

        if "occ_physical_properties" not in ABLATED_FEATURES:
            features_to_include.append(occurrence_area_scaled)
            features_to_include.append(occurrence_volume_scaled)

        if "visual_embeddings" not in ABLATED_FEATURES:
            features_to_include.append(visual_embeddings_scaled)

        if MATERIAL_TIER == 1:
            features_to_include.append(material_category_tier1)

        if MATERIAL_TIER == 2:
            features_to_include.append(material_category_tier1)
            features_to_include.append(material_category_tier2)

        if MATERIAL_TIER == 3:
            features_to_include.append(material_category_tier1)
            features_to_include.append(material_category_tier2)
            features_to_include.append(material_category_tier3)

        num_features = len(features_to_include)
        nodes = features_to_include.pop(0)

        for i in range(num_features - 1):
            feature = features_to_include.pop(0)
            nodes = torch.cat((nodes, feature), dim=-1)

        if len(features_to_include) != 0:
            print("Error in Ablation!")
            exit(1)

        material = torch.tensor([vocab[f'material'][n[-1][f'material']] for n in graph.nodes(data=True)])

        for edge in graph.edges(data=True):

            edges.append(torch.zeros(2 * len(vocab['edge_type']) + 1))

            if len(edge[-1]) == 0:
                edges[-1][0] = 1.
            else:
                if 'type' in edge[-1]:
                    edges[-1][vocab['edge_type'][edge[-1]['type']] + 1] = 1.

        edges = torch.stack(edges)
        edge_index = torch.tensor([[e[0], e[1]] for e in graph.edges()]).transpose(1, 0)

        processed_graphs.append(Data(x=nodes, edge_index=edge_index, e=edges, material=material, train_test=train_test))

    return processed_graphs, vocab, weights


class DataSet(object):
    def __init__(self, batch_size, downsample=1):

        self.graphs, self.vocab, self.weight = preprocess_data()

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.batch_size = batch_size
        self.downsample = downsample

        self.shuffle()
        self.node_dim = self.graphs[0].x.shape[-1]
        self.edge_dim = self.graphs[0].e.shape[-1]
        self.num_materials = len(self.vocab['material'])

    def shuffle(self):

        train_val_set, test_set = [], []
        for graph in self.graphs:
            if graph.train_test == "train":
                train_val_set.append(graph)
            elif graph.train_test == "test":
                test_set.append(graph)
            else:
                print("Error: incorrect train/test option!")
                exit(1)

        train, val = train_test_split(train_val_set, test_size=0.3, shuffle=True)

        if self.downsample < 1:
            print(f"Training size: {len(train)}")
            train = sample(train, round(self.downsample * len(train)))
            print(f"Training size downsampled: {len(train)}")
            assert len(train) > 0, print("Error: train/val set was down-sampled to 0")

        self.train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)


if __name__ == "__main__":
    dataset = DataSet(64)
