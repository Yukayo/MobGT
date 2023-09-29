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
        script_to_execute = "train_flashback.py"
        args1 = "toyota"
    ## Foursquare TKY Dataset
    elif dataset_type == "foursquare":
        script_to_execute = "train_flashback.py"
        args1 = "TKY-4sq"
    ## Gowalla Dataset
    elif dataset_type == "gowalla":
        script_to_execute = "train_flashback.py"
        args1 = "gowalla"
    else:
        raise ValueError("Unsupported dataset: {}, must be 'toyota', 'foursquare' or 'gowalla'")

    try:
        # Execute the Python script using the subprocess module
        subprocess.run(["python", script_to_execute, "--dataset", args1])
    except Exception as e:
        print(f"An error occurred: {e}")
    
if __name__ == "__main__":
    cli_main()