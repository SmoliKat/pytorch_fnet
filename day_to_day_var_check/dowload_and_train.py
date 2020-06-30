import argparse
import os
import json
from pathlib import Path
import datetime
import random
import quilt3
import pandas as pd
import secrets
import numpy as np
from fnet.cli.init import save_default_train_options
BASEDIR = ''

parser = argparse.ArgumentParser()

parser.add_argument("--gpu_id", default=0, type=int, help="GPU to use.")
parser.add_argument("--n_imgs", default=40, type=int, help="Number of images to use.")
parser.add_argument("--n_iterations", default=50000, type=int, help="Number of training iterations.")
parser.add_argument("--save_base_dir", default="/storage/users/assafzar/gil", type=str, help="savedir for images and models")
parser.add_argument("--experiment_name", default="gil_day_to_day_var", type=str, help="experiment name for tracking")
parser.add_argument(
    "--interval_checkpoint",
    default=10000,
    type=int,
    help="Number of training iterations between checkpoints.",
)

args = parser.parse_args()

###################################################
# Download the 3D multi-channel tiffs via Quilt/T4
###################################################

gpu_id = args.gpu_id
n_images_to_download = args.n_imgs  # more images the better
train_fraction = 0.75

image_save_dir = args.save_base_dir
base_path = os.path.join(args.save_base_dir, args.experiment_name)
model_save_dir = os.path.join(base_path, 'models')
prefs_save_path = os.path.join(base_path, 'prefs.json')

data_save_path_train = os.path.join(base_path, "image_list_train.csv")
data_save_path_test = os.path.join(base_path, "image_list_test.csv")

os.makedirs(image_save_dir, exist_ok=True)
os.makedirs(base_path, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)




aics_pipeline = quilt3.Package.browse(
    "aics/pipeline_integrated_cell", registry="s3://allencell"
)

data_manifest = aics_pipeline["metadata.csv"]()

# THE ROWS OF THE MANIFEST CORRESPOND TO CELLS, WE TRIM DOWN TO UNIQUIE FOVS
unique_fov_indices = np.unique(data_manifest['FOVId'], return_index=True)[1]
data_manifest = data_manifest.iloc[unique_fov_indices]

# SELECT THE FIRST N_IMAGES_TO_DOWNLOAD
#select images that match the Nuclear_envelope 
only_Endoplasmic_Reticulum = data_manifest.loc[data_manifest['StructureDisplayName']=='Endoplasmic reticulum']
only_Endoplasmic_Reticulum['date'] = only_Endoplasmic_Reticulum['SourceReadPath'].apply(lambda x:  datetime.datetime.strptime(x.split('_')[3], '%Y%m%d'))
n_train_images = int(n_images_to_download * train_fraction)
n_test_images = n_images_to_download - n_train_images
unique_dates = only_Endoplasmic_Reticulum['date'].unique()
train_date = secrets.choice(unique_dates)
test_date = secrets.choice(unique_dates)
print(f"train_date:{train_date} , test_date:{test_date}")

assert train_date != test_date, "train date and test date are the same date!"



X_train = only_Endoplasmic_Reticulum.loc[only_Endoplasmic_Reticulum.date == train_date].iloc[:n_train_images]
X_test = only_Endoplasmic_Reticulum.loc[only_Endoplasmic_Reticulum.date == test_date].iloc[:n_test_images]
data_manifest = pd.concat([X_train,X_test])
image_source_paths = data_manifest["SourceReadPath"]
image_target_paths = [
    "{}/{}".format(image_save_dir, image_source_path)
    for image_source_path in image_source_paths
]

for image_source_path, image_target_path in zip(image_source_paths, image_target_paths):
    if os.path.exists(image_target_path):
        continue

    # We only do this because T4 hates our filesystem. It probably wont affect you.
    try:
        aics_pipeline[image_source_path].fetch(image_target_path)
    except OSError:
        pass

###################################################
# Make a manifest of all of the files in csv form
###################################################

df = pd.DataFrame(columns=["path_tiff", "channel_signal", "channel_target"])

df["path_tiff"] = image_target_paths
df["channel_signal"] = data_manifest["ChannelNumberBrightfield"].values
#df["channel_signal_helper"] = data_manifest["ChannelNumber405"].values# this is the DNA channel for all FOVs, added as second chunnel for prediction
# df["channel_signal"] = [ [a,b] for a, b in zip(data_manifest["ChannelNumberBrightfield"], data_manifest["ChannelNumberBrightfield"])]
df["channel_target"] = data_manifest["ChannelNumberStruct"].values #change the chanel to be the structure.


df_train = df[:n_train_images]
df_test = df[n_train_images:]

df_test.to_csv(data_save_path_test, index=False)
df_train.to_csv(data_save_path_train, index=False)

################################################
# Run the label-free stuff (dont change this)
################################################

prefs_save_path = Path(prefs_save_path)

save_default_train_options(prefs_save_path)

#network_kwargs = {"depth": 4,
#"mult_chan": 32,
#"in_channels": 2,
#"out_channels": 1}

with open(prefs_save_path, "r") as fp:
    prefs = json.load(fp)

# takes about 16 hours, go up to 250,000 for full training
prefs["n_iter"] = int(args.n_iterations)
prefs["interval_checkpoint"] = int(args.interval_checkpoint)
#prefs["fnet_model_kwargs"]['nn_kwargs'] = network_kwargs #set the network kwargs to fit the 2 channel
prefs["dataset_train"] = "fnet.data.MultiChTiffDataset"
prefs["dataset_train_kwargs"] = {"path_csv": data_save_path_train}
prefs["dataset_val"] = "fnet.data.MultiChTiffDataset"
prefs["dataset_val_kwargs"] = {"path_csv": data_save_path_test}
prefs["bpds_kwargs"] = {'buffer_size': 16, 'buffer_switch_interval': 2800, 'patch_shape': [1, 32, 64, 64]}

# This Fnet call will be updated as a python API becomes available

with open(prefs_save_path, "w") as fp:
    json.dump(prefs, fp)

command_str = f"fnet train --json {prefs_save_path} --gpu_ids {gpu_id}"

print(command_str)
os.system(command_str)
