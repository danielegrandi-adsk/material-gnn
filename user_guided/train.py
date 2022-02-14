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
from user_guided.dataloader import DataSet, set_experiment, set_material_tier
from user_guided.GNN import MLP, Linear, CustomizedGNN
from user_guided.process_data import set_prediction
from sklearn.metrics import precision_score, recall_score, f1_score

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
            self.model = MLP(self.node_dim, self.hid_dim, self.num_materials).to(self.device)
        elif args.network == 'linear':
            self.model = Linear(self.node_dim, self.hid_dim, self.num_materials).to(self.device)
        else:
            self.model = CustomizedGNN(self.node_dim, self.edge_dim, self.hid_dim,
                                       self.num_materials,
                                       self.num_layers, args.network).to(self.device)

    def load(self):
        if os.path.exists('checkpoint.pkl'):
            self.model.load_state_dict(torch.load('checkpoint.pkl'))
        else:
            raise Exception('Checkpoint not found ...')

    def train(self, train_loader, val_loader, weights):

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
                batch = batch.to(self.device)
                optimizer.zero_grad()

                if args.network in ('mlp', 'linear'):
                    material_predictions = self.model(batch.x.float())
                else:
                    material_predictions = self.model(batch.x.float(), batch.edge_index, batch.e.float())

                loss = nn.CrossEntropyLoss(weight=weights_material)(material_predictions, batch.material)

                material_p.append(torch.argmax(material_predictions, dim=-1))
                material_t.append(batch.material)

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

            scheduler.step()
            end = time.time()
            val_loss, _, _, val_acc = self.predict(val_loader)

            material_p = torch.cat(material_p, -1)
            material_t = torch.cat(material_t, -1)

            train_acc = f1_score(y_true=material_t.cpu().numpy(),
                                 y_pred=material_p.cpu().numpy(), average="micro")

            if self.verbose:
                print(f'Epoch: {epoch + 1:03d}/{self.num_epochs} | Time: {end - start:.2f}s | '
                      f'Train Loss: {epoch_loss / len(train_loader):.4f} | Train Micro Acc: '
                      f'{100 * train_acc: .2f}% | Val Loss: {val_loss: .4f} | Val Micro Acc:'
                      f'{100 * val_acc: .2f}%')

            if best_loss > val_loss:
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
    def predict(self, data_loader):

        self.model.eval()
        loss = 0.
        material_p, material_t = [], []

        for batch in data_loader:
            batch = batch.to(self.device)

            if args.network in ('mlp', 'linear'):
                material_predictions = self.model.predict(batch.x.float())
            else:
                material_predictions = self.model.predict(batch.x.float(), batch.edge_index, batch.e.float())

            loss = nn.CrossEntropyLoss()(material_predictions, batch.material)

            material_p.append(torch.argmax(material_predictions, dim=-1))
            material_t.append(batch.material)

        loss /= len(data_loader)

        material_p = torch.cat(material_p, -1)
        material_t = torch.cat(material_t, -1)

        val_acc = f1_score(y_true=material_t.cpu().numpy(),
                           y_pred=material_p.cpu().numpy(), average="micro")

        return loss, material_p, material_t, val_acc

    @torch.no_grad()
    def predict_best_k(self, data_loader, best_k):

        self.model.eval()
        loss = 0.
        material_p, material_t = [], []

        for batch in data_loader:
            batch = batch.to(self.device)

            if args.network in ('mlp', 'linear'):
                material_predictions = self.model.predict(batch.x.float())
            else:
                material_predictions = self.model.predict(batch.x.float(), batch.edge_index, batch.e.float())

            loss = nn.CrossEntropyLoss()(material_predictions, batch.material)

            material_t.append(batch.material)

            if best_k == 1:
                predictions = torch.argmax(material_predictions, dim=-1)
                material_p.append(predictions)

            else:
                predictions = torch.topk(material_predictions, best_k).indices
                final_predictions = torch.argmax(material_predictions, dim=-1)

                for entry in range(len(predictions)):
                    if batch.material[entry] in predictions[entry]:
                        final_predictions[entry] = batch.material[entry]
                    else:
                        pass
                material_p.append(final_predictions)

        loss /= len(data_loader)

        material_p = torch.cat(material_p, -1)
        material_t = torch.cat(material_t, -1)
        raw_acc = f1_score(y_true=material_t.cpu().numpy(),
                           y_pred=material_p.cpu().numpy(), average="micro")

        return loss, material_p, material_t, raw_acc


def set_global_variables(experiment, prediction, topk, iteration, verbose=True, material_tier=0):
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
    set_material_tier(material_tier)


def calculate_mean(result):
    for measure in ['micro', 'macro', 'weighted']:
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
    return averaged_f1


def save_results(args, result, type="user_guided", method=None, class_tier=0):
    averaged_f1 = calculate_mean(result)

    if type == "user_guided":
        output_dir = Path("logs/User_Guided_Result.csv")

        if not output_dir.exists():
            df = pd.DataFrame(list(), columns=['Method', 'Hierarchy', 'Ablation', 'Prediction', 'Top K', 'Iteration',
                                               'Averaged Result', 'Std', 'Worst', 'Median', 'Best', 'Hash'])
            df.to_csv("logs/User_Guided_Result.csv")

        csv_row = [str(method), str(class_tier), str(ABLATION), str(PREDICTION), str(TOP_K), str(ITERATION),
                   str(round(result[f'material test results']['f1']['micro']['mean'], 4)),
                   str(round(result[f'material test results']['f1']['micro']['std'], 2)),
                   str(round(result[f'material test results']['f1']['micro']['min'], 4)),
                   str(round(result[f'material test results']['f1']['micro']['median'], 4)),
                   str(round(result[f'material test results']['f1']['micro']['max'], 4)),
                   str(hash(str(averaged_f1)))]

        dataframe = pd.DataFrame([csv_row],
                                 columns=['Method', 'Hierarchy', 'Ablation', 'Prediction', 'Top K', 'Iteration',
                                          'Averaged Result', 'Std', 'Worst', 'Median', 'Best', 'Hash'])
        dataframe.to_csv("logs/User_Guided_Result.csv", mode='a', header=False, index=True)

        if not os.path.exists('logs/JSON'):
            os.makedirs('logs/JSON')
        with open(f'logs/JSON/{hash(str(averaged_f1))}.json', 'w') as f:
            json.dump({**args.__dict__, **result}, f, indent=2)


def initialize_results():
    result = {
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


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='sage', choices=['gcn', 'gat', 'gin', 'sage', 'mlp', 'linear'])
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--hid_dim', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--num_layers', type=int, default=7)
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


def user_guided_experiment(args, method, topk, num_iter, class_tier, prediction, ablation):
    set_global_variables(experiment=ablation, prediction=prediction, topk=topk, iteration=num_iter, verbose=False,
                         material_tier=class_tier)

    # General parameters
    dataset = DataSet(args.batch_size, args.downsample)
    args.node_dim = dataset.node_dim
    args.edge_dim = dataset.edge_dim
    args.num_materials = dataset.num_materials

    # Hyper-parameters
    args.network = method
    args.hid_dim = 256
    args.num_layers = 7

    result = initialize_results()

    for __ in tqdm(range(num_iter), unit_scale=True, desc='Running User-guided Experiments...'):
        classifier = CustomizedClassifier(args)
        classifier.train(dataset.train_loader, dataset.val_loader, dataset.weight)

        loss, material_p, material_t, raw_acc = classifier.predict_best_k(dataset.test_loader, topk)

        material_p = material_p.cpu().numpy()
        material_t = material_t.cpu().numpy()

        for measure in ['micro', 'macro', 'weighted']:
            result[f'material test results']['f1'][measure]['data'].append(
                f1_score(y_true=material_t, y_pred=material_p, average=measure, zero_division=0))
            result[f'material test results']['precision'][measure]['data'].append(
                precision_score(y_true=material_t, y_pred=material_p, average=measure, zero_division=0))
            result[f'material test results']['recall'][measure]['data'].append(
                recall_score(y_true=material_t, y_pred=material_p, average=measure, zero_division=0))

        torch.cuda.empty_cache()

    save_results(args, result, type="user_guided", method=method, class_tier=class_tier)


if __name__ == '__main__':

    #########################################################
    # User-guided Experiment
    #########################################################

    print("Experiment: running [USER-GUIDED] experiment.")
    args = get_parser()

    methods = ["sage"]
    class_tiers = [
        0,  # None
        1,  # Tier 1
        2,  # Tier 1 + 2
        3,  # Tier 1 + 2 + 3
    ]
    topks = [3]

    for method in methods:
        for topk in topks:
            for class_tier in class_tiers:
                user_guided_experiment(args, method, topk,
                                       num_iter=15, class_tier=class_tier, prediction="material_id", ablation=[])

    print("Program finished normally.")
