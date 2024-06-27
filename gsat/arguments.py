import argparse


def load_gsat_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--n_layers', type = int, default = 2)
    parser.add_argument('--hidden_size', type = int, default = 64)
    parser.add_argument('--dropout_p', type = float, default = 0.3)
    parser.add_argument('--use_edge_attr', type = bool, default = False)
    parser.add_argument('--multi_label', type = bool, default = False)
    parser.add_argument('--learn_edge_att', type = bool, default = False)
    parser.add_argument('--extractor_dropout_p', type = float, default = 0.5)
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--weight_decay', type = float, default = 0)
    parser.add_argument('--precision_k', type = int, default = 5)
    parser.add_argument('--num_viz_samples', type = int, default = 16)
    parser.add_argument('--viz_interval', type = int, default = 10)
    parser.add_argument('--viz_norm_att', type = bool, default = True)
    parser.add_argument('--pred_loss_coef', type = float, default = 1)
    parser.add_argument('--info_loss_coef', type = float, default = 1)
    parser.add_argument('--fix_r', type = bool, default = False)
    parser.add_argument('--decay_r', type = float, default = 0.1)
    parser.add_argument('--decay_interval', type = int, default = 10)
    parser.add_argument('--final_r', type = float, default = 0.5)
    parser.add_argument('--init_r', type = float, default = 0.9)
    
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])
        
    return args
