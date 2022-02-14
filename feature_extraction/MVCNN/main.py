import os
import time
import csv
import torch
import torch.nn as nn
from mvcnn import Model
from args import get_parser
import torch.nn.functional as F
from dataset import MultiViewDataSet, preprocess
from torch.utils.data import DataLoader
# from helpers.logger import Logger
# import util
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# torch.use_deterministic_algorithms(True)
seed = 1
torch.manual_seed(seed)
import random
random.seed(seed)
np.random.seed(seed)


class Controller(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
        self.model = nn.DataParallel(Model(args.model, args.pretrained, args.emb_dim, args.n_class))
        self.model.to(self.device)

    def train(self, train_loader, val_loader):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epoch)
        best_acc, best_loss, patience_count, start_epoch = 0.0, 1e9, 0, 0
        weights = train_loader.dataset.weights.to(self.device)
        indices = torch.repeat_interleave(torch.arange(self.args.batch), self.args.views).to(self.device)
        # logger = Logger(self.args.model_path)

        if self.args.resume:
            best_acc, start_epoch, optimizer = self.load()

        for epoch in range(start_epoch, self.args.epoch):
            epoch_loss = .0
            total, correct = 0, 0
            start = time.time()
            self.model.train()

            for x, yt in train_loader:
                x, yt = x.to(self.device), yt.to(self.device)
                xi, xm, yp = self.model(x)
                if self.args.regime == 'supervised':
                    loss = Model.ce_loss(yp, yt, weights)
                elif self.args.regime == 'contrastive':
                    loss = Model.jsd_loss(xi, xm, indices)
                elif self.args.regime == 'hybrid':
                    loss = Model.ce_loss(yp, yt, weights) + Model.jsd_loss(xi, xm, indices)
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.args.regime != 'contrastive':
                    _, yp = torch.max(yp.data, 1)
                    total += yt.size(0)
                    correct += (yp == yt).sum().item()

            train_acc = 100 * correct / total if self.args.regime != 'contrastive' else .0

            end = time.time()
            self.model.eval()
            val_acc, val_loss = self.eval(val_loader)

            if self.args.regime != 'contrastive' and val_acc > best_acc:
                best_acc = val_acc
                if not os.path.exists(self.args.model_path):
                    os.mkdir(self.args.model_path)
                torch.save(self.model.module.state_dict(), f'{self.args.model_path}/model-best.pth')
                # torch.save(self.model.state_dict(), f'{self.args.model_path}/model-best.pth')

            print(f'Epoch {epoch + 1}/{self.args.epoch} | Time: {end - start:.2f}s '
                  f'| Train Loss: {epoch_loss / len(train_loader): .4f} | Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Val Acc: {val_acc:.2f}% | Best Acc: {best_acc:.2f}%')

            # Log epoch to tensorboard
            # See log using: tensorboard --logdir='args.model_path' --host localhost
            # util.logEpoch(logger, self.model, epoch + 1, val_loss, val_acc)

            if best_loss > val_loss:
                best_loss = val_loss
                patience_count = 0
                if self.args.regime == 'contrastive':
                    if not os.path.exists(self.args.model_path):
                        os.mkdir(self.args.model_path)
                    # torch.save(self.model.state_dict(), f'{self.args.model_path}/model-best.pth')
                    torch.save(self.model.module.state_dict(), f'{self.args.model_path}/model-best.pth')

            else:
                patience_count += 1

            if patience_count == self.args.patience:
                print(f'Early stopping at epoch {epoch} ...')
                break
            scheduler.step()

        # save model
        if not os.path.exists(self.args.model_path):
            os.mkdir(self.args.model_path)
        # torch.save(self.model.state_dict(), f'{self.args.model_path}/model-last.pth')
        torch.save(self.model.module.state_dict(), f'{self.args.model_path}/model-last.pth')

        # save labels
        labels = train_loader.dataset.classes
        with open(f'{self.args.model_path}/labels.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(labels)

        # print out evaluation report
        print("Validation report after training:")
        try:
            embeddings, predictions = self.encode(val_loader, self.args.model_path + "/model-last.pth")
            gt_classes, pred_classes = self.print_classification_report(val_loader, predictions)
        except Exception as e:
            print(e)

    @torch.no_grad()
    def eval(self, data_loader, load_model=False):
        weights = data_loader.dataset.weights.to(self.device)
        total, correct = 0, 0
        total_loss = 0.0

        if load_model:
            self.load()

        # test
        for x, yt in tqdm(data_loader, desc="Evaluating model"):
            x, yt = x.to(self.device), yt.to(self.device)
            xi, xm, yp = self.model(x)
            if self.args.regime == 'supervised':
                loss = Model.ce_loss(yp, yt, weights)
            elif self.args.regime == 'contrastive':
                indices = torch.repeat_interleave(torch.arange(x.size(0)), self.args.views).to(self.device)
                loss = Model.jsd_loss(xi, xm, indices)
            elif self.args.regime == 'hybrid':
                indices = torch.repeat_interleave(torch.arange(x.size(0)), self.args.views).to(self.device)
                loss = Model.ce_loss(yp, yt, weights) + Model.jsd_loss(xi, xm, indices)

            total_loss += loss.item()
            if self.args.regime != 'contrastive':
                _, yp = torch.max(yp.data, 1)
                total += yt.size(0)
                correct += (yp == yt).sum().item()

        val_acc = 100 * correct / total if self.args.regime != 'contrastive' else .0
        val_loss = total_loss / len(data_loader)
        return val_acc, val_loss

    @torch.no_grad()
    def encode(self, data_loader, model_path):
        try:
            self.model.load_state_dict(torch.load(model_path))
        except:
            state_dict = torch.load(model_path)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'module' not in k:
                    k = 'module.' + k
                else:
                    k = k.replace('features.module.', 'module.features.')
                new_state_dict[k] = v
            self.model.load_state_dict(new_state_dict)
        self.model.eval()
        emb, pred = [], []
        for x, __ in tqdm(data_loader, desc='Embedding...'):
            x = x.to(self.device)
            __, x, y = self.model(x)
            emb.append(x)
            pred.append(y)
        x = torch.cat(emb, 0).detach().cpu().numpy()
        y = F.softmax(torch.cat(pred, 0), dim=-1)
        return x, y

    def save_embeddings(self, data_loader, embs, classes):
        names = [Path(item).parts[-2] for item in data_loader.dataset.x]
        embedding_df = pd.DataFrame(list(zip(classes, names, embs)), columns=["class_name", "part_name", "vector"])

        dest = Path(self.args.model_path) / (Path(self.args.model_path).parts[-1] + '_embeddings')
        os.makedirs(dest, exist_ok=True)
        for class_name in tqdm(data_loader.dataset.classes, desc='Saving embeddings...'):
            class_embedding = embedding_df[embedding_df['class_name'] == class_name].to_numpy()
            np.save(dest / (class_name + "_embeddings"), class_embedding)

    def load(self): # Does not work
        print('\n==> Loading checkpoint..')
        model_path = self.args.model_path + "/model-last.pth"
        assert os.path.isfile(model_path), f'Error: no checkpoint file found in {model_path}!'
        checkpoint = torch.load(model_path)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
        return best_acc, start_epoch, optimizer

    def print_classification_report(self, encode_loader, predictions, top_k):
        import matplotlib.pyplot as plt
        import seaborn as sn

        gt_classes = [encode_loader.dataset.classes[item] for item in encode_loader.dataset.y]
        gt_classes_idx = [item for item in encode_loader.dataset.y]
        if top_k == 1:
            pred_classes_idx = np.argmax(predictions.detach().cpu().numpy(), axis=1)
            pred_classes = [encode_loader.dataset.classes[item] for item in pred_classes_idx]

            label = encode_loader.dataset.classes
            print(f"f1 micro precision: {f1_score(gt_classes, pred_classes, average='micro')}")
            print(classification_report(gt_classes, pred_classes, labels=label))

            cf = confusion_matrix(gt_classes, pred_classes, normalize='true', labels=label)

            if not os.path.exists('logs/'):
                os.makedirs('logs/')
            plt.figure(figsize=(24, 18))
            sn.heatmap(cf, annot=False, fmt='.2f', cmap='Blues', xticklabels=label, yticklabels=label)
            plt.xticks(size='xx-large', rotation=45)
            plt.yticks(size='xx-large', rotation=45)
            plt.tight_layout()
            plt.savefig(fname=f'logs/{Path(self.args.model_path).parts[-1]}.pdf', format='pdf')
            plt.show()
        else:
            log = pd.DataFrame(columns=['f1 micro'])
            for top in top_k:
                print(f"Calculating for top-{top}")
                final_predictions_idx = np.argmax(predictions.detach().cpu().numpy(), axis=1)
                top_predictions = torch.topk(predictions, top).indices
                for i in range(len(top_predictions)):
                    for pred in top_predictions[i]:
                        if pred == gt_classes_idx[i]:
                            final_predictions_idx[i] = pred

                pred_classes = [encode_loader.dataset.classes[item] for item in final_predictions_idx]
                label = encode_loader.dataset.classes

                f1_micro = f1_score(gt_classes, pred_classes, average='micro')
                print(f"Top-{top} f1 micro precision: {f1_micro}")
                # print(classification_report(gt_classes, pred_classes, labels=label))
                log = log.append(pd.Series({'f1 micro': f1_micro}, name=top))

                cf = confusion_matrix(gt_classes, pred_classes, normalize='true', labels=label)
                if not os.path.exists('logs/'):
                    os.makedirs('logs/')
                plt.figure(figsize=(24, 18))
                sn.heatmap(cf, annot=False, fmt='.2f', cmap='Blues', xticklabels=label, yticklabels=label)
                plt.xticks(size='xx-large', rotation=45)
                plt.yticks(size='xx-large', rotation=45)
                plt.tight_layout()
                plt.savefig(fname=f'logs/{Path(self.args.model_path).parts[-1]}_top-{top}.pdf', format='pdf')
                plt.show()
            log.to_csv(f"logs/{Path(self.args.model_path).parts[-1]}_mvcnn_topk.csv")

        return gt_classes, pred_classes


def main():
    args = get_parser()
    args.regime = 'contrastive'
    ds_train = MultiViewDataSet(args.data_path, ['train'])
    ds_val = MultiViewDataSet(args.data_path, ['test'])
    train_loader = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=8,
                              prefetch_factor=32, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=args.batch, shuffle=False, num_workers=8,
                            prefetch_factor=32, pin_memory=True)
    args.n_class = len(ds_train.classes)
    model = Controller(args)
    model.train(train_loader, val_loader)
    embeddings, predictions = model.encode(val_loader, args)


if __name__ == '__main__':
    main()
