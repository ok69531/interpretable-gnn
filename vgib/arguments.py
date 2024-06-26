import argparse

def load_vgib_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type = int, default = 10)
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument("--subgraph_const", type = float, default= 0.8 , help="Folder with training graph jsons.")
    parser.add_argument("--first-gcn-dimensions", type=int, default=16, help="Filters (neurons) in 1st convolution. Default is 32.")
    parser.add_argument("--second-gcn-dimensions", type=int, default=16, help="Filters (neurons) in 2nd convolution. Default is 16.")
    parser.add_argument("--first-dense-neurons", type=int, default=16, help="Neurons in SAGE aggregator layer. Default is 16.")
    parser.add_argument("--second-dense-neurons", type=int, default=2, help="SAGE attention neurons. Default is 8.")
    parser.add_argument("--cls_hidden_dimensions", type=int, default= 4, help="classifier hidden dims")
    parser.add_argument("--mi_weight", type=float, default= 0.1)
    parser.add_argument("--con_weight", type=float, default= 5)
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate. Default is 0.01.")
    parser.add_argument("--weight_decay", type=float, default=5*10**-5, help="Adam weight decay. Default is 5*10^-5.")

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])
    
    return args
