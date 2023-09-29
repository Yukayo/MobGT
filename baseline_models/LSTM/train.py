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
        script_to_execute = "train_toyota.py"
    ## Foursquare TKY Dataset
    elif dataset_type == "foursquare":
        script_to_execute = "train_foursquare.py"
    ## Gowalla Dataset
    elif dataset_type == "gowalla":
        script_to_execute = "train_gowalla.py"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_type}, must be 'toyota', 'foursquare' or 'gowalla'")
        
    cp_dir = os.path.join("exps", dataset_type, "checkpoint")
    os.makedirs(cp_dir, exist_ok=True)

    try:
        # Execute the Python script using the subprocess module
        subprocess.run(["python", script_to_execute])
    except Exception as e:
        print(f"An error occurred: {e}")
    
if __name__ == "__main__":
    cli_main()
