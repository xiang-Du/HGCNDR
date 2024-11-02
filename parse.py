import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="456")
    parser.add_argument('--gpu', action='store_true', help='enable gpu')
    parser.add_argument('--save_model', action='store_true', help='save_model')
    parser.add_argument('--variable_weight', action='store_true', help='enable varaiable_weight')
    parser.add_argument('--embed_dim1', type=int, default=64,
                        help="the embedding size entity and relation")
    parser.add_argument('--embed_dim2', type=int, default=64,
                        help="the embedding size entity and relation")
    parser.add_argument('--data_path', nargs='?', default='./data/Gdataset.mat',
                        help="data_path")  
    parser.add_argument('--mode', nargs='?', default='separated',
                        help="mode")
    parser.add_argument('--seed', type=int, default=1838)
    parser.add_argument('--valid_step', type=int, default=1)
    parser.add_argument('--layer_num', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=5000)
    parser.add_argument('--order', type=int, default=4)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--neg_ratio', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-4,
                        help="the learning rate")
    parser.add_argument('--rg', type=float, default=0.2,
                        help="the regulation rate")
    parser.add_argument('--decay', type=float, default=1e-4,
                        help="the weight decay for l2 regulation")
    parser.add_argument('--dropout', type=float, default=0.2,
                        help="using the dropout ratio")

    return parser.parse_args()
