import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'alexnet', 'vgg'])
    parser.add_argument('--regime', type=str, default='supervised', choices=['supervised', 'contrastive', 'hybrid'])
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--emb_dim', type=int, default=512)
    parser.add_argument('--views', type=int, default=12)
    parser.add_argument('--n_class', type=int)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--model_path', type=str, default='checkpoint')
    parser.add_argument('--top_k', type=str, default=1)
    parser.add_argument('--encode_train_test', type=str, default=['train', 'test'])
    parser.add_argument('--save_embeddings', type=str, default=True)
    return parser.parse_args()

