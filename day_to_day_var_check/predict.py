import os
import argparse

###################################################
# Assume the user already ran download_and_train.py
###################################################

parser = argparse.ArgumentParser()

parser.add_argument("--gpu_id", default=0, type=int, help="GPU to use.")
parser.add_argument("--save_base_dir", default="/storage/users/assafzar/gil", type=str, help="savedir for images and models")
parser.add_argument("--experiment_name", default="gil_day_to_day_var", type=str, help="experiment name for tracking")
args = parser.parse_args()

# Normally this would be run via command-line but this Fnet call will be updated as a python API becomes available
gpu_id = args.gpu_id

image_save_dir = args.save_base_dir
model_save_dir = os.path.join(args.save_base_dir, args.experiment_name)
predictions_save_dir = os.path.join(args.save_base_dir, args.experiment_name, 'predictions')
os.makedirs(predictions_save_dir, exist_ok=True)

# image_save_dir = "{}/images/".format(os.getcwd())
# model_save_dir = "{}/model/".format(os.getcwd())
data_save_path_test = os.path.join(image_save_dir, args.experiment_name, "image_list_test.csv")

# data_save_path_test = "{}/image_list_test.csv".format(os.getcwd())

# command_str = (
#     f"""fnet predict --path_model_dir {model_save_dir} --dataset fnet.data.MultiChTiffDataset  --dataset_kwargs \'{{"path_csv": "{data_save_path_test}"}}\' --gpu_ids {gpu_id} """
# )
command_str = (
    "fnet predict "
    "--path_model_dir {} "
    "--dataset fnet.data.MultiChTiffDataset "
    '--dataset_kwargs \'{{"path_csv": "{}"}}\' '
    "--gpu_ids {} "
    "--path_save_dir {}".format(model_save_dir, data_save_path_test, gpu_id,predictions_save_dir)
)

print(command_str)
os.system(command_str)
