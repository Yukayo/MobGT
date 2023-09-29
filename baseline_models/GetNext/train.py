import os
from argparse import ArgumentParser
import subprocess


def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--dataset_type", "-dt", help="Dataset type to be processed")
    args = parser.parse_args()
    dataset_type = args.dataset_type
    
    if dataset_type == "toyota":
        script_to_execute = "train_caller.py"
        data_adj_mtx = "../../dataset/baseline_models_dataset/GetNext/toyota/graph_A.csv"
        data_node_feats = "../../dataset/baseline_models_dataset/GetNext/toyota/graph_X.csv"
        data_train = "../../dataset/baseline_models_dataset/GetNext/toyota/train.csv"
        data_val = "../../dataset/baseline_models_dataset/GetNext/toyota/val.csv"
        data_test = "../../dataset/baseline_models_dataset/GetNext/toyota/val.csv"
    ## Foursquare TKY Dataset
    elif dataset_type == "foursquare":
        script_to_execute = "train_caller.py"
        data_adj_mtx = "../../dataset/baseline_models_dataset/GetNext/tky/graph_A.csv"
        data_node_feats = "../../dataset/baseline_models_dataset/GetNext/tky/graph_X.csv"
        data_train = "../../dataset/baseline_models_dataset/GetNext/tky/train.csv"
        data_val = "../../dataset/baseline_models_dataset/GetNext/tky/val.csv"
        data_test = "../../dataset/baseline_models_dataset/GetNext/tky/val.csv"
    ## Gowalla Dataset
    elif dataset_type == "gowalla":
        script_to_execute = "train_caller.py"
        data_adj_mtx = "../../dataset/baseline_models_dataset/GetNext/gowalla_nevda/graph_A.csv"
        data_node_feats = "../../dataset/baseline_models_dataset/GetNext/gowalla_nevda/graph_X.csv"
        data_train = "../../dataset/baseline_models_dataset/GetNext/gowalla_nevda/train.csv"
        data_val = "../../dataset/baseline_models_dataset/GetNext/gowalla_nevda/val.csv"
        data_test = "../../dataset/baseline_models_dataset/GetNext/gowalla_nevda/val.csv"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_type}, must be 'toyota', 'foursquare' or 'gowalla'")
    
    cp_dir = os.path.join("exps", dataset_type, "checkpoint")
    os.makedirs(cp_dir, exist_ok=True)

    try:
        # Execute the Python script using the subprocess module
        subprocess.run([
            "python", script_to_execute, 
            "--data-adj-mtx", data_adj_mtx, 
            "--data-node-feats", data_node_feats,
            "--data-train", data_train,
            "--data-val", data_val,
            "--data-test", data_test,
            "--dataset_type", dataset_type,
            "--save_dir", dataset_type,
        ])
    except Exception as e:
        print(f"An error occurred: {e}")
    
if __name__ == "__main__":
    cli_main()
