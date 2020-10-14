import os
import sys
import numpy as np
import pickle
import pandas as pd
from glob import glob
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from logging import getLogger

logger = getLogger("downloader")
BASEPATH = "/storage/users/assafzar/experiments/day_to_day/"


def get_exp_variables_from_folder(exp_file):
    split_train = exp_file.split("_train_")
    exp_struct = split_train[0]
    split_test = split_train[1].split("_test_")
    test = split_test[1]
    train_dates = split_test[0].split("_")
    return {"structure":exp_struct, "test_date":test, "train_dates":train_dates}


def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes

# def parser():
    # parser = argparse.ArgumentParser(prog="fnet")
    # parser.add_argument("-s", "--image-save-dir", type=Path, help="save dir", default="/storage/users/assafzar", )
    # parser.add_argument("--base_exp_path", type=Path, help="save path for exp",
    #                     default="/storage/users/assafzar/experiments/day_to_day", )
    # parser.add_argument("--gpu_id", default=0, type=int, help="GPU to use.")
    # parser.add_argument("--n_imgs", default=40, type=int, help="Number of images to use.")
    # parser.add_argument("--n_iterations", default=40000, type=int, help="Number of training iterations.")
    # parser.add_argument("-n", "--num-images-download", type=int, help="how many images to download", default=4)
    # parser.add_argument(
    #     "--interval_checkpoint",
    #     default=10000,
    #     type=int,
    #     help="Number of training iterations between checkpoints.",
    # )
    # args = parser.parse_args()
    # return args
def expand_dict_with_test(dictionary,test_data_date,values, train_data_list):
    if test_data_date in dictionary:
        if len(train_data_list)>3:
            dictionary[test_data_date]['x']= values
        else:
            dictionary[test_data_date]['y']=values
    else:
        dictionary[test_data_date] = {}
        if len(train_data_list)>3:
            dictionary[test_data_date]['x']=values
        else:
            dictionary[test_data_date]['y']=values




def main():
    sys.path.append(os.getcwd())
    # parser()
    if not os.path.isfile('data.p'):
        result = [y for x in os.walk(BASEPATH) for y in glob(os.path.join(x[0], 'predictions.csv'))]
        list_exp_points = {}
        for exp_path in result:
            df = pd.read_csv(exp_path)
            column_name = None
            for col in df.columns:
                if 'corr_coef' in col:
                    column_name = col
            dir_name = os.path.dirname(os.path.dirname(exp_path))
            exp = os.path.split(dir_name)[1]
            exp_metadata = get_exp_variables_from_folder(exp)
            list_exp_points[exp] = {'values': df[column_name].values, 'mean': df[column_name].mean()}
            list_exp_points[exp].update(exp_metadata)
        with open('data.p', 'wb') as fp:
            pickle.dump(list_exp_points, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('data.p', 'rb') as fp:
            list_exp_points = pickle.load(fp)
    print(list_exp_points)
    # random_dict = list_exp_points.pop(x)

    training_losses = [y for x in os.walk(BASEPATH) for y in glob(os.path.join(x[0], 'losses.csv'))]
    for loss_path in training_losses:
        test = pd.read_csv(loss_path)
        num_iter, loss_train, los_val = test.min()
        print(test.loc[test['loss_train'] == loss_train])






    dictionary = {}
    for key, dict_value in list_exp_points.items():
        if dict_value['structure'] in dictionary:
            expand_dict_with_test(dictionary[dict_value['structure']], dict_value['test_date'], dict_value['values'], dict_value['train_dates'])
            # dictionary[dict_value['structure']] = np.concatenate((dictionary[dict_value['structure']],dict_value['values']),axis=0)
            # dictionary[dict_value['structure']] = dict_value['test_date']
        else:
            dictionary[dict_value['structure']] = {}
            # dictionary[dict_value['structure']] = dict_value['values']
            expand_dict_with_test(dictionary[dict_value['structure']], dict_value['test_date'], dict_value['values'], dict_value['train_dates'])
    # dictionary = {dict_value['structure']:dict_value['values'] for (key, dict_value) in list_exp_points.items()}
    dictionary.pop('Microtubules')

    vals, names, xs = [], [], []
    for key, value in dictionary.items():
        names.append(key)
        per_strcuture_vals = np.array([])
        per_strucutre_xs = np.array([])
        for test_date, values in value.items():
            per_strcuture_vals  = np.concatenate((per_strcuture_vals,values['y']),axis=0)
            per_strucutre_xs = np.concatenate((per_strucutre_xs, values['x']), axis=0)
            # per_strucutre_xs = per_strucutre_xs + values['x']
            # per_strucutre_xs.append(values['x'])

        vals.append(per_strcuture_vals)
        xs.append(per_strucutre_xs)

    plt.figure(figsize=(15, 15))
    plt.xlabel('random_day_to_day')
    plt.ylabel('training')


    plt.boxplot(vals, labels=names)
    palette = ['r', 'g', 'b', 'y']
    for x, val, c in zip(xs, vals, palette):
        plt.scatter(x, val, alpha=0.4, color=c)
    plt.xticks(rotation=90)
    plt.savefig('data.png')

    plt.show()
    # plt.title(random_dict['structure'])
    ax = sns.boxplot(x="structure", y="structure", data=df, showfliers=False)
    # ax = sns.stripplot(x=f"{x_name}", y="all_training", hue="train_dates",
    #                      data=df, style="train_dates", s=100)
    # ax.set(ylim=(0.5, 0.8))
    # ax.set(xlim=(0.5, 0.8))

#
# plt.figure(figsize=(10,10))
# plt.xlabel('random_day_to_day')
# plt.ylabel('training')
# plt.title(random_dict['structure'])
# ax = sns.scatterplot(x=f"{x_name}", y="all_training", hue="train_dates",
#                      data=df,style="train_dates", s=100)
# add_identity(ax, color='r', ls='--')
# ax.set(ylim=(0.5, 0.8))
# ax.set(xlim=(0.5, 0.8))

if __name__ == "__main__":
    main()
