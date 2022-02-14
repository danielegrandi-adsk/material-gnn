from torch.utils.data import DataLoader

from args import get_parser
from dataset import MultiViewDataSet, preprocess
from main import Controller

import torch
torch.manual_seed(42)
import random
random.seed(42)
import numpy as np
np.random.seed(42)

################################################
arg1 = get_parser()
arg1.model = 'resnet'
arg1.epoch = 100
arg1.batch = 8
arg1.regime = 'supervised'
arg1.training_data_path = [r"FusionGallery_training"]
arg1.encoding_data_path = [r"FusionGallery_encoding"]
arg1.model_path = r'model_path'
arg1.training = True
arg1.save_embeddings = True
arg1.encode_train_test = ['test']
arg1.top_k = [1, 2, 3]
################################################


def main():
    models_to_run = [arg1]   # Modify this to train/embed different models

    for args in models_to_run:
        print(f"Working with model: {args.model_path}")

        print("Preprocessing...")
        preprocess(args.data_path, args.views)

        if args.training:
            print(f"Loading data from: {args.training_data_path}")
            ds_train = MultiViewDataSet(args.training_data_path, ['train'])
            ds_val = MultiViewDataSet(args.training_data_path, ['test'])
            train_loader = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=16,
                                      prefetch_factor=32, pin_memory=True, drop_last=True)
            val_loader = DataLoader(ds_val, batch_size=args.batch, shuffle=False, num_workers=16,
                                    prefetch_factor=32, pin_memory=True)
            args.n_class = len(ds_train.classes)
            model = Controller(args)

            print("Training model...")
            model.train(train_loader, val_loader)

        print(f"Encoding {args.encode_train_test} data from: {args.encoding_data_path}")
        ds_encode = MultiViewDataSet(args.encoding_data_path, args.encode_train_test) #, args.model_path)
        encode_loader = DataLoader(ds_encode, batch_size=args.batch, shuffle=False, num_workers=16,
                                   prefetch_factor=32, pin_memory=True)
        args.n_class = len(ds_encode.classes)
        model = Controller(args)

        embeddings, predictions = model.encode(encode_loader, args.model_path + "/model-last.pth")
        print(f"Classification report of {args.encode_train_test}")
        gt_classes, pred_classes = model.print_classification_report(encode_loader, predictions, args.top_k)

        if args.save_embeddings:
            print("Saving embeddings")
            model.save_embeddings(encode_loader, embeddings, gt_classes)


if __name__ == '__main__':
    main()
