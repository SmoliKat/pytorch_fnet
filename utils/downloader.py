import argparse
import os
import sys
from logging import getLogger
from pathlib import Path
import numpy as np
import quilt3
from aicsimageio import AICSImage
import tifffile
logger = getLogger("downloader")


def transform_image_into_channel_volumes(image_path, req_channels):
    # aicsimageio.imread loads as STCZYX, so we load only CZYX
    im_out = {}
    with AICSImage(image_path) as img:
        im_tmp = img.get_image_data("CZYX", S=0, T=0)

    for req_channel in req_channels:
        im_out[f'channel_{req_channel}'] = (im_tmp[req_channel])

    return im_out
    # im_out.append(im_tmp[req_channels])


def downloader(aics_pipeline, image_source_path, image_target_path):
    save_dir = os.path.dirname(image_target_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(image_target_path):
        try:
            aics_pipeline[image_source_path].fetch(image_target_path)
            return True
        except OSError:
            logger.error(f"could not download file {image_source_path}, {str(OSError)}")
            return False
    else:
        return True

    # image_target_paths = [
    #     "{}/{}".format(image_save_dir, image_source_path)
    #     for image_source_path in image_source_paths
    # ]
    #
    # for image_source_path, image_target_path in zip(image_source_paths, image_target_paths):


def read_csv(csv_path):
    raise NotImplementedError


def write_results_to_csv(results, csv_path):
    raise NotImplementedError

def channel_field_names_to_numbers(data_manifest, channels_to_save):
    results = []
    for channel_to_save in channels_to_save:
        results.append(data_manifest[channel_to_save].values[0])
    return results



def main():
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser(prog="fnet")
    parser.add_argument(
        "-s", "--save-dir", type=Path, help="save dir", default="/storage/users/assafzar/fovs",
    )
    parser.add_argument("-n", "--num-images-download", type=int, help="how many images to download", default=1)
    parser.add_argument("--structure-display-name", help="structure display name to download", required=False)
    parser.add_argument("--channels-to-save", type=str, nargs='+', help="channels to save", default=["ChannelNumberBrightfield"])
    args = parser.parse_args()

    aics_pipeline = quilt3.Package.browse(
        "aics/pipeline_integrated_cell", registry="s3://allencell"
    )
    data_manifest = aics_pipeline["metadata.csv"]()
    unique_fov_indices = np.unique(data_manifest['FOVId'], return_index=True)[1]
    data_manifest = data_manifest.iloc[unique_fov_indices]

    # SELECT THE FIRST N_IMAGES_TO_DOWNLOAD
    # select images that match the Nuclear_envelope
    only_needed_data = data_manifest.loc[data_manifest['StructureDisplayName'] == 'Endoplasmic reticulum']
    data_manifest = only_needed_data.iloc[0:args.num_images_download]
    image_source_paths = data_manifest["SourceReadPath"]
    image_target_paths = [
        os.path.join(args.save_dir, image_source_path)
        for image_source_path in image_source_paths
    ]

    channels_to_save = channel_field_names_to_numbers(data_manifest=data_manifest, channels_to_save=args.channels_to_save)

    for image_source_path, image_target_path in zip(image_source_paths, image_target_paths):
        download_ok = downloader(aics_pipeline=aics_pipeline, image_source_path=image_source_path,
                                 image_target_path=image_target_path)
        if download_ok:
            channel_volumes_dict = transform_image_into_channel_volumes(image_target_path, channels_to_save)
            for channel_key, channel_volume in channel_volumes_dict.items():
                save_path = os.path.join(args.save_dir, 'Endoplasmic reticulum',os.path.splitext(os.path.basename(image_source_path))[0])
                os.makedirs(save_path, exist_ok=True)
                file_path = os.path.join(save_path,f'{channel_key}.tif')
                tifffile.imsave(file_path, channel_volume, compress=2)
                #np.savez_compressed(os.path.join(save_path,f'{channel_key}.npz'), volume=channel_volume)
        else:
            logger.error(f"error encountered during download {image_source_path}")


if __name__ == "__main__":
    main()
