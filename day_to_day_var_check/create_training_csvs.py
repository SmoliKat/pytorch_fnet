import argparse
import datetime
import json
import os
import sys
from logging import getLogger
from pathlib import Path
import numpy as np
import quilt3
from aicsimageio import AICSImage
import tifffile
import pandas as pd
from fnet.cli.init import save_default_train_options

logger = getLogger("downloader")

aics_pipeline = quilt3.Package.browse(
    "aics/pipeline_integrated_cell", registry="s3://allencell"
)


def get_exp_name(structure, pattern):
    exp_name = f'{structure}_train'
    for train_date in pattern["train"]:
        exp_name = f"{exp_name}_{train_date}"
    exp_name = f"{exp_name}_test_{pattern['test']}"
    return exp_name


def get_download_path(data_manifest, image_save_dir, structure_display_name):
    image_source_paths = data_manifest["SourceReadPath"]
    image_target_paths = [
        os.path.join(image_save_dir, 'fovs', structure_display_name, os.path.basename(image_source_path))
        for image_source_path in image_source_paths
    ]
    for image_source_path, image_target_path in zip(image_source_paths, image_target_paths):
        if os.path.exists(image_target_path):
            continue

        try:
            aics_pipeline[image_source_path].fetch(image_target_path)
        except OSError:
            pass

    return image_target_paths


def save_csvs(data_manifest, exp_name, base_exp_path, filename, image_save_dir, structure_display_name):
    df = pd.DataFrame(columns=["path_tiff", "channel_signal", "channel_target"])

    df["path_tiff"] = get_download_path(data_manifest=data_manifest, image_save_dir=image_save_dir,
                                        structure_display_name=structure_display_name)
    df["channel_signal"] = data_manifest["ChannelNumberBrightfield"].values
    # df["channel_signal_helper"] = data_manifest["ChannelNumber405"].values# this is the DNA channel for all FOVs, added as second chunnel for prediction
    # df["channel_signal"] = [ [a,b] for a, b in zip(data_manifest["ChannelNumberBrightfield"], data_manifest["ChannelNumberBrightfield"])]
    df["channel_target"] = data_manifest["ChannelNumberStruct"].values  # change the chanel to be the structure.
    save_path = os.path.join(base_exp_path, exp_name)
    os.makedirs(save_path, exist_ok=True)
    save_csv_path = os.path.join(save_path, filename)
    if not (os.path.exists(save_csv_path)):
        df.to_csv(save_csv_path, index=False)
    return save_csv_path


def create_slurm_runner(base_dir,exp_list,prefs_file_name='prefs.json'):
    with open("exp_runner.sh", "w") as file_object:
        for exp in exp_list:
            running_line = f'sbatch train_mk2.sh "{os.path.join(base_dir,exp,prefs_file_name)}" "{exp}"'
            file_object.write(f"echo {running_line}\n")
            file_object.write(f"{running_line}\n")
        file_object.close()


def main():
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser(prog="fnet")
    parser.add_argument("-s", "--image-save-dir", type=Path, help="save dir", default="/storage/users/assafzar", )
    parser.add_argument("--base_exp_path", type=Path, help="save path for exp",
                        default="/storage/users/assafzar/experiments/day_to_day", )
    parser.add_argument("--gpu_id", default=0, type=int, help="GPU to use.")
    parser.add_argument("--n_imgs", default=40, type=int, help="Number of images to use.")
    parser.add_argument("--n_iterations", default=40000, type=int, help="Number of training iterations.")
    parser.add_argument("-n", "--num-images-download", type=int, help="how many images to download", default=4)
    parser.add_argument(
        "--interval_checkpoint",
        default=10000,
        type=int,
        help="Number of training iterations between checkpoints.",
    )
    args = parser.parse_args()

    whole_data_manifest = aics_pipeline["metadata.csv"]()

    unique_fov_indices = np.unique(whole_data_manifest['FOVId'], return_index=True)[1]
    whole_data_manifest = whole_data_manifest.iloc[unique_fov_indices]
    whole_data_manifest['date'] = whole_data_manifest['SourceReadPath'].apply(
        lambda x: datetime.datetime.strptime((x.split('_')[3]).split('-')[0], '%Y%m%d').strftime('%d-%m-%Y'))
    unique_structures = whole_data_manifest['Structure'].unique()
    n_images_to_download = args.n_imgs  # more images the better
    train_fraction = 0.75
    n_train_images = int(n_images_to_download * train_fraction)
    n_test_images = n_images_to_download - n_train_images
    exp_list = []
    for structure_display_name in unique_structures:
        only_needed_data = whole_data_manifest.loc[whole_data_manifest['Structure'] == structure_display_name]
        unique_dates = only_needed_data['date'].unique()
        if len(unique_dates)<4:
            print(f"issue with unique dates length for {structure_display_name}")
            continue

        print(
            f"FOR {structure_display_name} these unique dates : {unique_dates} and have {[{'date': unique_date, 'items': len(only_needed_data.loc[only_needed_data.date == unique_date])} for unique_date in unique_dates]}")
        date_range = unique_dates[:4]
        patterns = [{"train": [date_range[0], date_range[1], date_range[2]], "test": date_range[3]},
                    {"train": [date_range[0], date_range[0], date_range[0]], "test": date_range[3]},
                    {"train": [date_range[1], date_range[1], date_range[1]], "test": date_range[3]},
                    {"train": [date_range[2], date_range[2], date_range[2]], "test": date_range[3]},
                    {"train": [date_range[0], date_range[1], date_range[2], date_range[3]], "test": date_range[3]}]

        # df_to_download = pd.DataFrame(columns=[col for col in only_needed_data.columns])
        # first get test set
        X_test = pd.DataFrame(columns=[col for col in only_needed_data.columns])
        X_train = pd.DataFrame(columns=[col for col in only_needed_data.columns])

        date_to_test = date_range[3]
        if len(only_needed_data.loc[only_needed_data.date == date_to_test]) > n_test_images:
            for index in range(0, n_test_images):
                item = only_needed_data.loc[only_needed_data.date == date_to_test].iloc[:1]
                X_test = X_test.append(item)
                only_needed_data.drop(
                    only_needed_data.index[only_needed_data.index.tolist().index(item.index[0])], inplace=True)
        else:
            raise (Exception(
                f"not enough images only {len(only_needed_data.loc[only_needed_data.date == date_to_test])} for date {date_to_test}"))
        # create X_train
        for pattern in patterns:
            counter = 0
            while counter < n_train_images:
                for date in pattern["train"]:
                    item = only_needed_data.loc[only_needed_data.date == date].iloc[:1]
                    if item is not None:
                        X_train = X_train.append(item)
                        counter += 1
                        only_needed_data.drop(
                            only_needed_data.index[only_needed_data.index.tolist().index(item.index[0])])
            exp_name = get_exp_name(structure_display_name, pattern)
            train_csv_path = save_csvs(data_manifest=X_train, exp_name=exp_name, base_exp_path=args.base_exp_path,
                      filename="image_list_train.csv", image_save_dir=args.image_save_dir,
                      structure_display_name=structure_display_name)
            test_csv_path = save_csvs(data_manifest=X_test, exp_name=exp_name, base_exp_path=args.base_exp_path,
                      filename="image_list_test.csv", image_save_dir=args.image_save_dir,
                      structure_display_name=structure_display_name)

            #pref file
            prefs_save_path = os.path.join(args.base_exp_path, exp_name, 'prefs.json')
            if not (os.path.exists(prefs_save_path)):
                prefs_save_path = Path(prefs_save_path)

                save_default_train_options(prefs_save_path)

                # network_kwargs = {"depth": 4,
                # "mult_chan": 32,
                # "in_channels": 2,
                # "out_channels": 1}
                with open(prefs_save_path, "r") as fp:
                    prefs = json.load(fp)

                # takes about 16 hours, go up to 250,000 for full training
                prefs["n_iter"] = int(args.n_iterations)
                prefs["interval_checkpoint"] = int(args.interval_checkpoint)
                # prefs["fnet_model_kwargs"]['nn_kwargs'] = network_kwargs #set the network kwargs to fit the 2 channel
                prefs["dataset_train"] = "fnet.data.MultiChTiffDataset"
                prefs["dataset_train_kwargs"] = {"path_csv": train_csv_path}
                prefs["dataset_val"] = "fnet.data.MultiChTiffDataset"
                prefs["dataset_val_kwargs"] = {"path_csv": test_csv_path}
                prefs["bpds_kwargs"] = {'buffer_size': 16, 'buffer_switch_interval': 2800, 'patch_shape': [1, 32, 64, 64]}

                with open(prefs_save_path, "w") as fp:
                    json.dump(prefs, fp)
            exp_list.append(exp_name)
    create_slurm_runner(args.base_exp_path,exp_list)



if __name__ == "__main__":
    main()
