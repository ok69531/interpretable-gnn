import argparse


def load_gib_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type = int, default = 2)
    parser.add_argument('--hidden', type = int, default = 16)
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--lr_decay_factor', type = float, default = 0.5)
    parser.add_argument('--lr_decay_step_size', type = int, default = 50)
    parser.add_argument('--inner_loop', type = int, default = 50)
    parser.add_argument('--beta', type = float, default = 0.1)
    parser.add_argument('--pp_weight', type = float, default = 0.3)

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])
    
    return args