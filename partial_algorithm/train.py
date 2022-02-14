import os
import time
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import numpy as np
import json
import random
from pathlib import Path
import pandas as pd
from partial_algorithm.dataloader import DataSet, set_experiment, set_lazy_loading
from partial_algorithm.GNN import MLP, Linear, CustomizedGNN
from partial_algorithm.process_data import set_prediction
from torch.nn import functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

ABLATION, PREDICTION, ITERATION, TOP_K = None, None, None, None

"""Reproducibility"""
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class CustomizedClassifier(object):
    def __init__(self, args):
        self.verbose = args.verbose
        self.device = torch.device(args.device)
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.num_layers = args.num_layers

        self.num_materials = args.num_materials

        self.patience = args.patience
        self.node_dim = args.node_dim
        self.edge_dim = args.edge_dim
        self.hid_dim = args.hid_dim
        self.lr = args.lr

        if args.network == 'mlp':
            self.model = MLP(self.node_dim, self.hid_dim, self.num_class_l1,
                             self.num_class_l2, self.num_class_l3).to(self.device)
        elif args.network == 'linear':
            self.model = Linear(self.node_dim, self.hid_dim, self.num_class_l1,
                                self.num_class_l2, self.num_class_l3).to(self.device)
        else:  # Default is here

            self.model = CustomizedGNN(self.node_dim, self.edge_dim, self.hid_dim,
                                       self.num_materials,
                                       self.num_layers, args.network).to(self.device)

    def load(self):
        if os.path.exists('checkpoint.pkl'):
            self.model.load_state_dict(torch.load('checkpoint.pkl'))
        else:
            raise Exception('Checkpoint not found ...')

    def train(self, train_loader, val_loader, weights, mask_amount, masked_val_nodes, permuted):
        best_loss, best_state, patience_count = 1e9, self.model.state_dict(), 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)

        weights_material = weights[0].to(self.device)

        for epoch in range(self.num_epochs):
            material_p, material_t = [], []

            self.model.train()
            epoch_loss = 0.
            start = time.time()

            for batch in train_loader:
                if not permuted:
                    materials_feature = F.one_hot(batch.material, self.num_materials)  # one-hot material vector
                    random_mask = np.random.choice([0, 1], size=len(batch.x), p=[1 - mask_amount,
                                                                                 mask_amount])  # Masked=0 -> node does not get extra material info
                    materials_feature = materials_feature * random_mask[:,
                                                            None]  # set materials_feature to zeros if masked
                    batch.x = torch.cat((batch.x, materials_feature), dim=1)  # add materials_feature to batch.x

                batch = batch.to(self.device)
                optimizer.zero_grad()

                if args.network in ('mlp', 'linear'):
                    logits_l1, logits_l2, logits_l3 = self.model(
                        batch.x.float(), F.one_hot(batch.y1, self.num_class_l1), F.one_hot(batch.y2, self.num_class_l2))
                else:  # by default

                    material_predictions = self.model(batch.x.float(), batch.edge_index, batch.e.float())

                # compute loss (with ground truth)
                if not permuted:
                    is_masked = torch.tensor([True if mask == 0 else False for mask in
                                              random_mask])  # true if the node did not see get any more info about its label
                else:
                    is_masked = torch.tensor([True if node_mask == 0 else False for graph in batch.mask for node_mask in
                                          graph])  # true if the node did not  get any more info about its label
                loss = nn.CrossEntropyLoss(weight=weights_material)(material_predictions[is_masked],
                                                                    batch.material[
                                                                        is_masked])  # only include in the loss nodes that are not masked
                # loss = nn.CrossEntropyLoss(weight=weights_material)(material_predictions, batch.material)

                # only append if not masked
                material_p.append(torch.argmax(material_predictions[is_masked], dim=-1))
                material_t.append(batch.material[is_masked])

                epoch_loss += loss.item()
                loss.backward()  # back propagation (compute gradients and update parameters)
                optimizer.step()  # optimizer takes one step

            scheduler.step()  # scheduler takes one step
            end = time.time()
            val_loss, _, _, val_acc = self.predict(val_loader,
                                                   weights_material, masked_val_nodes,
                                                   mask_amount, permuted)  # evaluate and obtain loss on validation set

            material_p = torch.cat(material_p, -1)
            material_t = torch.cat(material_t, -1)

            train_acc = f1_score(y_true=material_t.cpu().numpy(),
                                 y_pred=material_p.cpu().numpy(), average="micro")

            if self.verbose:
                print(f'Epoch: {epoch + 1:03d}/{self.num_epochs} | Time: {end - start:.2f}s | '
                      f'Train Loss: {epoch_loss / len(train_loader):.4f} | Train Micro Acc: '
                      f'{100 * train_acc: .2f}% | Val Loss: {val_loss: .4f} | Val Micro Acc:'
                      f'{100 * val_acc: .2f}%')

            if best_loss > val_loss:  # if this state better than previous best, store it
                best_loss = val_loss
                best_state = self.model.state_dict()
                patience_count = 0
            else:
                patience_count += 1

            if patience_count == self.patience:
                if self.verbose:
                    print('Early stopping ...')
                break

        self.model.load_state_dict(best_state)
        print("Saving the model...")
        torch.save(best_state, 'checkpoint.pkl')

    @torch.no_grad()
    def predict(self, data_loader, weights_material, masked_val_nodes,
                mask_amount, permuted):  # data_loader is the testing/validation set

        self.model.eval()  # set to evaluation
        loss = 0.
        material_p, material_t = [], []

        num_nodes = 0
        for batch in data_loader:
            if not permuted:
                # Pad with actual material info for nodes that are not masked, taking the mask from the dataset
                materials_feature = F.one_hot(batch.material, self.num_materials)  # one-hot material vector
                random_mask = masked_val_nodes[
                              num_nodes: num_nodes + len(batch.x)]  # Masked=0 -> node does not get extra material info
                try:
                    materials_feature = materials_feature * random_mask[:,
                                                            None]  # set materials_feature to zeros if masked
                except:
                    print()
                batch.x = torch.cat((batch.x, materials_feature), dim=1)

            batch = batch.to(self.device)
            # x = node attributes; edge_index = 2 nodes to each edge; e = edge attributes
            if args.network in ('mlp', 'linear'):  # ignore this
                logits_l1, logits_l2, logits_l3 = self.model.predict(batch.x.float())
            else:
                material_predictions = self.model.predict(batch.x.float(), batch.edge_index, batch.e.float())

            if not permuted:
                is_masked = torch.tensor([True if mask == 0 else False for mask in
                                          random_mask])  # true if the node did not see get any more info about its label
            else:
                is_masked = torch.tensor([True if node_mask == 0 else False for graph in batch.mask for node_mask in
                                          graph])  # true if the node did not see get any more info about its label
            loss = nn.CrossEntropyLoss()(material_predictions[is_masked], batch.material[
                is_masked])  # only include in the loss nodes that are not masked

            # only append if not masked
            material_p.append(torch.argmax(material_predictions[is_masked], dim=-1))
            material_t.append(batch.material[is_masked])

            num_nodes += len(batch.x)

        loss /= len(data_loader)

        material_p = torch.cat(material_p, -1)
        material_t = torch.cat(material_t, -1)

        val_acc = f1_score(y_true=material_t.cpu().numpy(),
                           y_pred=material_p.cpu().numpy(), average="micro")

        return loss, material_p, material_t, val_acc

    @torch.no_grad()
    def predict_best_K(self, data_loader, weights_material, best_k, mask_amount, masked_test_nodes, permuted):

        self.model.eval()  # set to evaluation
        loss = 0.
        material_p, material_t = [], []

        batch_num = 0
        for batch in data_loader:
            if not permuted:
                # Pad with actual material info for nodes that are not masked, taking the mask from the dataset
                materials_feature = F.one_hot(batch.material, self.num_materials)  # one-hot material vector
                random_mask = masked_test_nodes[batch_num * len(batch.x): (batch_num + 1) * len(
                    batch.x)]  # Masked=0 -> node does not get extra material info
                materials_feature = materials_feature * random_mask[:, None]  # set materials_feature to zeros if masked
                batch.x = torch.cat((batch.x, materials_feature), dim=1)

            batch = batch.to(self.device)
            # x = node attributes; edge_index = 2 nodes to each edge; e = edge attributes
            if args.network in ('mlp', 'linear'):  # ignore this
                logits_l1, logits_l2, logits_l3 = self.model.predict(batch.x.float())
            else:
                material_predictions = self.model.predict(batch.x.float(), batch.edge_index, batch.e.float())

            if not permuted:
                is_masked = torch.tensor([True if mask == 0 else False for mask in
                                          random_mask])  # true if the node did not see get any more info about its label
            else:
                is_masked = torch.tensor([True if node_mask == 0 else False for graph in batch.mask for node_mask in
                                          graph])  # true if the node did not see get any more info about its label
            loss = nn.CrossEntropyLoss()(material_predictions[is_masked], batch.material[
                is_masked])  # only include in the loss nodes that are not masked

            # only append if not masked
            material_t.append(batch.material[is_masked])

            # Top K predictions

            if best_k == 1:
                material_p.append(torch.argmax(material_predictions[is_masked], dim=-1))
            else:
                predictions = torch.topk(material_predictions[is_masked], best_k).indices
                final_predictions = torch.argmax(material_predictions[is_masked], dim=-1)

                for entry in range(len(predictions)):
                    if batch.material[entry] in predictions[entry]:
                        final_predictions[entry] = batch.material[entry]
                    else:
                        pass
                material_p.append(final_predictions)

            batch_num += 1

        loss /= len(data_loader)

        material_p = torch.cat(material_p, -1)
        material_t = torch.cat(material_t, -1)
        raw_acc = f1_score(y_true=material_t.cpu().numpy(),
                           y_pred=material_p.cpu().numpy(), average="micro")

        return loss, material_p, material_t, raw_acc


def set_global_variables(experiment, prediction, topk, iteration, verbose=True):
    global ABLATION, PREDICTION, TOP_K, ITERATION

    if prediction not in ["material_id", "material_category_tier_1", "material_category_full"]:
        print("ERROR: invalid prediction choice!")
        exit(1)

    PREDICTION = prediction
    TOP_K = topk
    ITERATION = iteration
    ABLATION = experiment
    if verbose:
        print(f"Prediction: {PREDICTION} | Top K: {TOP_K} | Iteration: {ITERATION} | Ablation: {experiment}.")

    set_prediction(prediction)
    set_experiment(experiment)


def save_results(args, result):
    for measure in ['micro', 'macro', 'weighted']:  # calculate mean and std across all iterations
        result[f'material test results']['f1'][measure]['max'] = np.max(
            result[f'material test results']['f1'][measure]['data'])
        result[f'material test results']['f1'][measure]['mean'] = np.mean(
            result[f'material test results']['f1'][measure]['data'])
        result[f'material test results']['f1'][measure]['median'] = np.median(
            result[f'material test results']['f1'][measure]['data'])
        result[f'material test results']['f1'][measure]['min'] = np.min(
            result[f'material test results']['f1'][measure]['data'])
        result[f'material test results']['f1'][measure]['std'] = np.std(
            result[f'material test results']['f1'][measure]['data'])
        result[f'material test results']['precision'][measure]['mean'] = np.mean(
            result[f'material test results']['precision'][measure]['data'])
        result[f'material test results']['precision'][measure]['std'] = np.std(
            result[f'material test results']['precision'][measure]['data'])
        result[f'material test results']['recall'][measure]['mean'] = np.mean(
            result[f'material test results']['recall'][measure]['data'])
        result[f'material test results']['recall'][measure]['std'] = np.std(
            result[f'material test results']['recall'][measure]['data'])
    if not os.path.exists('logs/'):
        os.makedirs('logs/')
    averaged_f1 = result[f'material test results']['f1']['micro']['mean']

    output_dir = Path("logs/Training_Result.csv")

    if not output_dir.exists():
        df = pd.DataFrame(list(), columns=['Down-sample', 'Ablation', 'Prediction',
                                           'Top K', 'Iteration', 'Averaged Result', 'Std', 'Worst', 'Median', 'Best',
                                           'Hash'])
        df.to_csv("logs/Training_Result.csv")

    csv_row = [str(args.downsample), str(ABLATION), str(PREDICTION), str(TOP_K), str(ITERATION),
               str(round(result[f'material test results']['f1']['micro']['mean'], 4)),
               str(round(result[f'material test results']['f1']['micro']['std'], 2)),
               str(round(result[f'material test results']['f1']['micro']['min'], 4)),
               str(round(result[f'material test results']['f1']['micro']['median'], 4)),
               str(round(result[f'material test results']['f1']['micro']['max'], 4)),
               str(hash(str(averaged_f1)))]

    dataframe = pd.DataFrame([csv_row], columns=['Down-sample', 'Ablation', 'Prediction',
                                                 'Top K', 'Iteration', 'Averaged Result', 'Std', 'Worst', 'Median',
                                                 'Best', 'Hash'])

    dataframe.to_csv("logs/Training_Result.csv", mode='a', header=False, index=True)

    if not os.path.exists('logs/JSON'):
        os.makedirs('logs/JSON')

    with open(f'logs/JSON/{hash(str(averaged_f1))}.json', 'w') as f:
        json.dump({**args.__dict__, **result}, f, indent=2)


def save_results_tuning(args, result, network, num_layers, hid_dim, mask_amount):
    for measure in ['micro', 'macro', 'weighted']:  # calculate mean and std across all iterations
        result[f'material test results']['f1'][measure]['max'] = np.max(
            result[f'material test results']['f1'][measure]['data'])
        result[f'material test results']['f1'][measure]['mean'] = np.mean(
            result[f'material test results']['f1'][measure]['data'])
        result[f'material test results']['f1'][measure]['median'] = np.median(
            result[f'material test results']['f1'][measure]['data'])
        result[f'material test results']['f1'][measure]['min'] = np.min(
            result[f'material test results']['f1'][measure]['data'])
        result[f'material test results']['f1'][measure]['std'] = np.std(
            result[f'material test results']['f1'][measure]['data'])
        result[f'material test results']['precision'][measure]['mean'] = np.mean(
            result[f'material test results']['precision'][measure]['data'])
        result[f'material test results']['precision'][measure]['std'] = np.std(
            result[f'material test results']['precision'][measure]['data'])
        result[f'material test results']['recall'][measure]['mean'] = np.mean(
            result[f'material test results']['recall'][measure]['data'])
        result[f'material test results']['recall'][measure]['std'] = np.std(
            result[f'material test results']['recall'][measure]['data'])

    if not os.path.exists('logs/'):
        os.makedirs('logs/')

    averaged_f1 = result[f'material test results']['f1']['micro']['mean']

    output_dir = Path("logs/Tuning_Result.csv")

    columns = ['Network', 'Num_layers', 'Hidden_dim',
               'Averaged Result', 'Std', 'Worst', 'Median', 'Best', 'mask_amount', 'Hash']
    if not output_dir.exists():
        df = pd.DataFrame(list(), columns=columns)
        df.to_csv("logs/Tuning_Result.csv")

    csv_row = [str(network), str(num_layers), str(hid_dim),
               str(round(result[f'material test results']['f1']['micro']['mean'], 4)),
               str(round(result[f'material test results']['f1']['micro']['std'], 2)),
               str(round(result[f'material test results']['f1']['micro']['min'], 4)),
               str(round(result[f'material test results']['f1']['micro']['median'], 4)),
               str(round(result[f'material test results']['f1']['micro']['max'], 4)),
               str(mask_amount),
               str(hash(str(averaged_f1)))]

    dataframe = pd.DataFrame([csv_row], columns=columns)

    dataframe.to_csv("logs/Tuning_Result.csv", mode='a', header=False, index=True)

    if not os.path.exists('logs/JSON'):
        os.makedirs('logs/JSON')
    with open(f'logs/JSON/{hash(str(averaged_f1))}.json', 'w') as f:  # save the test results
        json.dump({**args.__dict__, **result}, f, indent=2)


def initialize_results():
    result = {  # Store the results to be put into the test log
        'material test results': {
            'f1': {
                'macro': {'mean': .0, 'std': .0, 'data': []},
                'micro': {'mean': .0, 'std': .0, 'data': []},
                'weighted': {'mean': .0, 'std': .0, 'data': []},
            },
            'precision': {
                'macro': {'mean': .0, 'std': .0, 'data': []},
                'micro': {'mean': .0, 'std': .0, 'data': []},
                'weighted': {'mean': .0, 'std': .0, 'data': []},
            },
            'recall': {
                'macro': {'mean': .0, 'std': .0, 'data': []},
                'micro': {'mean': .0, 'std': .0, 'data': []},
                'weighted': {'mean': .0, 'std': .0, 'data': []},
            }
        }
    }
    return result


def partial_experiment(args, mask_amount, network, num_layers, hidden_dim, num_iter, dataset=None, permuted=False):
    print(f"Partial algorithm-guided experiment: "
          f"masking amount = {mask_amount} | network = {network} | num_layers = {num_layers} | hidden_dim = {hidden_dim}")

    # General parameters
    if not permuted:
        set_global_variables(experiment=[], prediction="material_id", topk=1, iteration=num_iter, verbose=False)
        dataset = DataSet(args.batch_size, args.downsample, mask_amount)
        args.node_dim = dataset.node_dim + dataset.num_materials
    else:
        args.node_dim = dataset.node_dim
    args.edge_dim = dataset.edge_dim
    args.num_materials = dataset.num_materials

    # Hyper-parameters
    args.network = network
    args.hid_dim = hidden_dim
    args.num_layers = num_layers

    result = initialize_results()

    for __ in tqdm(range(num_iter), unit_scale=True, desc='Running partial masking amount experiments...'):
        classifier = CustomizedClassifier(args)
        classifier.train(dataset.train_loader, dataset.val_loader, dataset.weight,
                         mask_amount, dataset.masked_val_nodes, permuted)  # Train on train dataset

        weights_material = dataset.weight[0].to(args.device)  # Initialized material weights

        loss, material_p, material_t, raw_acc = classifier.predict_best_K(dataset.test_loader, weights_material, TOP_K,
                                                                          mask_amount, dataset.masked_test_nodes, permuted)

        material_p = material_p.cpu().numpy()
        material_t = material_t.cpu().numpy()

        for measure in ['micro', 'macro', 'weighted']:
            result[f'material test results']['f1'][measure]['data'].append(
                f1_score(y_true=material_t, y_pred=material_p, average=measure))
            result[f'material test results']['precision'][measure]['data'].append(
                precision_score(y_true=material_t, y_pred=material_p, average=measure))
            result[f'material test results']['recall'][measure]['data'].append(
                recall_score(y_true=material_t, y_pred=material_p, average=measure))

        torch.cuda.empty_cache()
        print(classification_report(y_true=material_t, y_pred=material_p, digits=3))

    save_results_tuning(args, result, network, num_layers, hidden_dim, mask_amount)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='sage', choices=['gcn', 'gat', 'gin', 'sage', 'mlp', 'linear'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--hid_dim', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--downsample', type=float, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--node_dim', type=int)
    parser.add_argument('--edge_dim', type=int)
    parser.add_argument('--num_class_l1', type=int)
    parser.add_argument('--num_class_l2', type=int)
    parser.add_argument('--num_class_l3', type=int)
    parser.add_argument('--verbose', type=bool, default=True)
    return parser.parse_args()


def partial_permuted_experiments(args, amounts, network=["sage"], layers=[1], hidden_dim=[256], num_iter=5):
    set_global_variables(experiment=[], prediction="material_id", topk=1, iteration=num_iter, verbose=False)

    for amount in amounts:
        dataset = DataSet(args.batch_size, args.downsample, amount, True)
        for net in network:
            for layer in layers:
                for dim in hidden_dim:
                    partial_experiment(args, amount, net, layer, dim, num_iter, dataset, permuted=True)


if __name__ == '__main__':
    args = get_parser()

    #########################################################
    # Partial algorithm-guided prediction, masking all nodes:
    #########################################################
    print("Experiment: partial algorithm-guided prediction, masking all nodes.")

    masking_amounts = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1]
    layers = [1, 2, 3, 4, 5, 6, 7, 8]

    partial_permuted_experiments(args, masking_amounts, layers=layers, num_iter=1)

    print("Program finished normally.")
