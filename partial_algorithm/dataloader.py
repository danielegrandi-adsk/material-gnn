import numpy as np
import _pickle as cPickle
import bz2
from partial_algorithm.process_data import *
import torch.nn as nn
import torch
import networkx as nx
from collections import Counter
from torch_geometric.data import DataLoader, Data
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tqdm import tqdm
from partial_algorithm.config import ABLATED_FEATURES, GLOBAL_FEATURES
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


def set_lazy_loading():
    global LAZY_LOADING
    LAZY_LOADING = True


def get_vocab():
    """Get the vocab dictionary, weights dictionary, and list of NetworkX graphs"""

    graphs, input_files = assembly2graph()  # List of NetworkX graphs

    # Only include the feature(s) that we want to predict:
    material_count = []

    # Vocab dictionary

    vocab = {
        # 'depth_of_hierarchy': set(),  # one for each global assembly, currently not used

        # Node features
        'material': set(),
        'appearance_id': set(),
        'appearance_name': set(),
        'body_name': set(),
        'body_type': set(),
        'occurrence_name': set(),
        'material_category_tier1': set(),

        # Edge features
        'edge_type': set(),
        'edge_weight': set(),

        # Global features
        'products': set(),
        'categories': set(),
        'industries': set(),
    }

    for input_file in tqdm(input_files, desc="Preprocessing Graphs"):

        ag = AssemblyGraph(input_file)  # Initialize a class for this file

        if not ORIGINAL_EDGE_FEATURES:
            nodes, _, edges, depth, train_test = ag.get_graph_data()
        else:
            nodes, edges, _, depth, train_test = ag.get_graph_data()

        for node in nodes:
            vocab['material'].add(node['material'])
            vocab['appearance_id'].add(node['appearance_id'])
            vocab['appearance_name'].add(node['appearance_name'])
            vocab['material_category_tier1'].add(node['material_category']['tier1'])
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
                if not ORIGINAL_EDGE_FEATURES:
                    vocab['edge_type'].add(edge['type'])
                    vocab['edge_weight'].add(edge['edge_weight'])
                else:
                    vocab['edge_type'].add(edge['type'])

    for k, v in vocab.items():
        vocab[k] = {s: idx for idx, s in enumerate(sorted(v))}

    material_count = [vocab['material'][l] for l in material_count]

    # Initialize weights for material:
    material_w = [0.] * len(vocab['material'])

    # k = index of instance, v = number of appearance of this instance
    # len(material_count) = total number of samples,
    # Weight[i] = n_samples / (n_classes) * (n_i)

    for k, v in Counter(material_count).items():
        material_w[k] = len(material_count) / v  # Original
        # material_w[k] = v / len(material_count)  # Original flipped

        # Weight[i] = n_samples / (n_classes) * (n_i)
        # material_w[k] = len(material_count) / (len(vocab["material"]) * v)

    weights = {0: torch.tensor(material_w)}  # Should be balanced weights

    return graphs, vocab, weights


def preprocess_data(masked=False, permuted=False, mask_amount=1):
    processed_graphs = []
    data, vocab, weights = get_vocab()
    graph_num, node_num, edge_num = 0, 0, 0

    for graph in tqdm(data, desc="Encoding Features"):

        if graph.number_of_nodes() < 3 or graph.number_of_edges() < 2:  # graph too small, pass
            continue
        # if graph.number_of_nodes() <= 0 or graph.number_of_edges() <= 0:  # graph too small, pass
        #     continue
        train_test = None
        for node in graph.nodes(data=True):
            train_test = node[-1]["train_test"]  # Obtain whether this graph is for train or test
            break

        nodes, edges = [], []

        graph_num += 1
        node_num += graph.number_of_nodes()
        edge_num += graph.number_of_edges()

        # relabel index/uid into corresponding strings/names
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

            # Squeeze the dimension from [X, 1, 60] to [X, 60]
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

        if not ORIGINAL_EDGE_FEATURES:
            """INTEGER features: edge_weight"""
            edge_weight_tensor = torch.tensor([[e[-1]['edge_weight']] for e in graph.edges(data=True)],
                                              dtype=torch.float)

            edge_weight_scaler = preprocessing.StandardScaler().fit(edge_weight_tensor)
            edge_weight_scaled = edge_weight_scaler.transform(edge_weight_tensor)
            edge_weight_scaled = torch.from_numpy(edge_weight_scaled)

        material_category_tier1 = nn.functional.one_hot(torch.tensor(
            [vocab['material_category_tier1'][n[-1]['material_category_tier1']] for n in graph.nodes(data=True)]),
            len(vocab['material_category_tier1']))

        if len(ABLATED_FEATURES) == 0:  # No ablation
            nodes = torch.cat((

                # material_category_tier1, # Warning: including tier 1 material category may be cheating

                # Center of mass
                center_x_scaled,
                center_y_scaled,
                center_z_scaled,

                # Tech-Net encoding of names:

                body_name_embeddings_scaled,
                occ_name_embeddings_scaled,

                # Encode the physical properties:

                body_area_scaled,
                body_volume_scaled,

                occurrence_area_scaled,
                occurrence_volume_scaled,

                # Visual embeddings:

                visual_embeddings_scaled,

                global_features_scaled

            ), -1)

        else:  # Select the features not ablated

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

            num_features = len(features_to_include)
            nodes = features_to_include.pop(0)

            for i in range(num_features - 1):  # One already popped
                feature = features_to_include.pop(0)
                nodes = torch.cat((nodes, feature), dim=-1)

            if len(features_to_include) != 0:
                print("Error in Ablation!")
                exit(1)

        # Encoding of Material ground truths:

        material = torch.tensor([vocab[f'material'][n[-1][f'material']] for n in graph.nodes(data=True)])

        # One-hot encoding of EDGE features:

        for edge in graph.edges(data=True):

            edges.append(torch.zeros(2 * len(vocab['edge_type']) + 1))  # list of tensors

            if len(edge[-1]) == 0:  # no feature
                edges[-1][0] = 1.
            else:
                if 'type' in edge[-1]:
                    edges[-1][vocab['edge_type'][edge[-1]['type']] + 1] = 1.

        edges = torch.stack(edges)  # concatenate torch tensors, dim=0 by default

        if not ORIGINAL_EDGE_FEATURES:
            edges = torch.cat((edges, edge_weight_tensor), -1)

        edge_index = torch.tensor([[e[0], e[1]] for e in graph.edges()]).transpose(1, 0)  # each edge 2 node indexes

        if not permuted and mask_amount < 1:
            processed_graphs.append(Data(x=nodes, edge_index=edge_index, e=edges, material=material,
                                         train_test=train_test))
        elif mask_amount < 1 and permuted:
            materials_feature = nn.functional.one_hot(material, len(vocab['material']))  # one-hot material vector
            for i in range(int(len(nodes) * mask_amount)):
                # create new graph with: new data in embedding, extra 'masked' array with node masking info
                mask = np.random.choice([1, 0], size=len(nodes), p=[mask_amount, 1 - mask_amount])  # Masked=0 -> node does not get extra material info

                masked_materials_feature = materials_feature * mask[:, None]  # set materials_feature to zeros if mask == 0
                nodes_with_material = torch.cat((nodes, masked_materials_feature),
                                                dim=1)  # add materials_feature to batch.x

                processed_graphs.append(Data(x=nodes_with_material, edge_index=edge_index, e=edges, material=material,
                                             train_test=train_test, mask=mask))
        elif mask_amount == 1:
            materials_feature = nn.functional.one_hot(material, len(vocab['material']))  # one-hot material vector
            for i in range(len(nodes)):
                # create new graph with: new data in embedding, extra 'masked' array with node masking info
                mask = np.ones(len(nodes))
                mask[i] = 0  # Mask=0 -> node does not get extra material info
                masked_materials_feature = materials_feature * mask[:, None]  # set materials_feature to zeros if masked
                nodes_with_material = torch.cat((nodes, masked_materials_feature), dim=1)  # add materials_feature to batch.x

                processed_graphs.append(Data(x=nodes_with_material, edge_index=edge_index, e=edges, material=material,
                                             train_test=train_test, non_masked_node=i, mask=mask))

    print(f"Masking amount: {mask_amount}")
    print("Number of permuted graphs: ", len(processed_graphs))
    print("Number of original graphs: ", graph_num)
    return processed_graphs, vocab, weights


def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        cPickle.dump(data, f)


def decompress_pickle(file):
 data = bz2.BZ2File(file, 'rb')
 data = cPickle.load(data)
 return data


class DataSet(object):
    def __init__(self, batch_size, downsample=1, mask_amount=0, permuted=False):

        if not LAZY_LOADING and downsample == 1 and mask_amount == 0:
            self.graphs, self.vocab, self.weight = preprocess_data()
        elif mask_amount == 1 or permuted:
            self.graphs, self.vocab, self.weight = preprocess_data(permuted=True, mask_amount=mask_amount)
            # dataset_path = f"{DATA_PATH}/dataset_materialID_TOP20_masked.pkl"
            #
            # if os.path.exists(dataset_path):
            #     print(f"Loading data directly from {dataset_path}")
            #     # with open(dataset_path, 'rb') as inp:
            #     #     self.graphs, self.vocab, self.weight = pickle.load(inp)
            #     self.graphs, self.vocab, self.weight = decompress_pickle(dataset_path)
            # else:
            #     self.graphs, self.vocab, self.weight = preprocess_data(masked=True)
            #     # with open(dataset_path, 'wb') as outp:
            #     #     pickle.dump([self.graphs, self.vocab, self.weight], outp)
            #     compressed_pickle(dataset_path, [self.graphs, self.vocab, self.weight])

        elif mask_amount != 0 and permuted == False:
            dataset_path = f"{DATA_PATH}/dataset_materialID_TOP20_fullFeatures_sharedOcc.pkl"
            if os.path.exists(dataset_path):
                print(f"Loading data directly from {dataset_path}")
                with open(dataset_path, 'rb') as inp:
                    self.graphs, self.vocab, self.weight = pickle.load(inp)
            else:
                self.graphs, self.vocab, self.weight = preprocess_data(mask_amount=mask_amount)
                with open(dataset_path, 'wb') as outp:
                    pickle.dump([self.graphs, self.vocab, self.weight], outp)

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.batch_size = batch_size
        self.downsample = downsample
        self.mask_amount = mask_amount

        self.shuffle()
        self.node_dim = self.graphs[0].x.shape[-1]
        self.edge_dim = self.graphs[0].e.shape[-1]
        self.num_materials = len(self.vocab['material'])

    def shuffle(self):
        # Train: Val: Test = 6 : 1 : 3
        # train, test_val = train_test_split(self.graphs, test_size=0.4, shuffle=True)
        # val, test = train_test_split(test_val, test_size=0.75, shuffle=True)

        # Train: Val: Test = 7 : 1.5 : 1.5
        # train, test_val = train_test_split(self.graphs, test_size=0.3, shuffle=True)
        # val, test = train_test_split(test_val, test_size=0.5, shuffle=True)

        train_val_set, test_set = [], []
        for graph in self.graphs:
            if graph.train_test == "train":
                train_val_set.append(graph)
            elif graph.train_test == "test":
                test_set.append(graph)
            else:
                print("Error: incorrect train/test option!")
                exit(1)
        # Train+val : Test = 8 : 2
        train, val = train_test_split(train_val_set, test_size=0.3, shuffle=True)  # Train : val = 7:3

        if self.downsample < 1:
            print(f"Training size: {len(train)}")
            train = sample(train, round(self.downsample * len(train)))
            print(f"Training size downsampled: {len(train)}")
            assert len(train) > 0, print("Error: train/val set was down-sampled to 0")

        self.train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        if self.mask_amount < 1:
            # list of randomly picked nodes for val and test that will be masked
            self.masked_val_nodes = np.random.choice([1, 0], size=sum([graph.num_nodes for graph in val]),
                                                     p=[1 - self.mask_amount,
                                                        self.mask_amount])
            # print(len(self.masked_val_nodes))
            # print(sum([graph.num_nodes for graph in val]))
            self.masked_test_nodes = np.random.choice([1, 0], size=sum([graph.num_nodes for graph in test_set]),
                                                      p=[1 - self.mask_amount, self.mask_amount])
            print(f"Number of train nodes that do not see extra material info and are included in the loss: "
                  f"{self.mask_amount * sum([graph.num_nodes for graph in train])} / {sum([graph.num_nodes for graph in train])}")
            print(f"Number of val nodes that do not see extra material info and are included in the loss: "
                  f"{self.mask_amount * sum([graph.num_nodes for graph in val])} / {sum([graph.num_nodes for graph in val])}")
            print(f"Number of test nodes that do not see extra material info and are included in the loss: "
                  f"{self.mask_amount * sum([graph.num_nodes for graph in test_set])} / {sum([graph.num_nodes for graph in test_set])}")
        else:
            self.masked_val_nodes = np.array([graph.non_masked_node for graph in self.val_loader.dataset])
            self.masked_test_nodes = np.array([graph.non_masked_node for graph in self.test_loader.dataset])


if __name__ == "__main__":
    dataset = DataSet(64)
