from fnet.cli import train_model
import os
import sys
import argparse
from pathlib import Path

def main(): 
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser(prog="fnet")
    parser.add_argument(
            "--json", type=Path,  help="json with training options", default="/storage/users/assafzar/experiments/day_to_day/Microtubules_train_01-03-2017_01-03-2017_01-03-2017_test_03-03-2017/prefs.json"
        )
    parser.add_argument("--gpu_ids", nargs="+", default=[0], type=int, help="gpu_id(s)")
    args = parser.parse_args()
    train_model.main(args)

if __name__ == "__main__":
    main()
#fnet train --json /home/kathrine/pytorch_fnet/model/prefs.json --gpu_ids 0