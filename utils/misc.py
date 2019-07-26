import logging
import numpy
import uproot
import json
import os


def load_config(path):
    """
    Load configuration file with all the needed parameters
    """
    with open(path, 'r') as conf_file:
        conf = json.load(conf_file)
    return conf


def get_logger(name):
    """
    Add a StreamHandler to a logger if still not added and
    return the logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.propagate = 1  # propagate to parent
        console = logging.StreamHandler()
        logger.addHandler(console)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
    return logger


utils_log = get_logger(__name__)
utils_log.setLevel(logging.INFO)


def load_data(rootfile_path, df_columns):
    """
    Loading dataframe using uproot.tree method.
    Cleaning infinity and nan, and dropping
    duplicates.
    :param rootfile_path: path/to/file.root
    :param df_columns: full list of variables
                       (feat + lab)
    :return: features, dataset
    """
    f = uproot.open(rootfile_path)
    tree = f['Clustering;1']
    dataset = tree.pandas.df(df_columns)
    utils_log.info("before : {}".format(dataset.shape))
    features = dataset.loc[:, dataset.columns].values
    dataset = dataset.replace([numpy.inf, -numpy.inf], numpy.nan).dropna(how='any')
    dataset = dataset.drop_duplicates()
    utils_log.info("after checking nan and inf : {}".format(dataset.shape))
    return features, dataset

def print_df_info(df):
    """
    Print main info dataset
    :param df: dataset
    :return: None
    """
    n_merge_max = df["merge_vtx"].max()
    n_merge_mean = df["merge_vtx"].mean()
    n_evt_max = df["event_numb"].max()
    utils_log.info("############### USEFUL DATASET INFOS #####################")
    utils_log.info("dataset columns: {}".format(list(df.columns)))
    utils_log.info("dataset shape: {}".format(df.shape))
    utils_log.info("nEvt: {}".format(n_evt_max))
    utils_log.info("MAX Number of merged vertices per event: {}".format(n_merge_max))
    utils_log.info("AVERAGE Number of merged vertices per event: {}".format(n_merge_mean))
    utils_log.info("##########################################################")

def plot_direc_creator(path_to_directory, evt, mrg, n_trks):
    """
    Create a directory for each event where to store the plots.
    :param path_to_directory: path/to/directory
    :param evt: event string
    :param mrg: merge vertex string
    :param n_trks: number of tracks
    :return:
    """
    event_dir = os.path.join(path_to_directory, "Plots",
                             "evt_" + str(evt))
    plot_out_dir = os.path.join(event_dir, 'MrgVtx' +
                                str(mrg) + '_nTrks' +
                                str(n_trks))
    if not os.path.exists(plot_out_dir):
        os.makedirs(plot_out_dir)
    return plot_out_dir

def tracks_csv_creator(folder, df1, df_hs, df_pu, n_trks):
    """
    Filling csv files with the track's features values.
    :param folder: path/to/save/file
    :param df1: full dataframe
    :param df_hs: HS dataframe
    :param df_pu: PU datagrame
    :param n_trks: number of tracks
    :return: None
    """
    if folder is not None:
        df1_path = os.path.join(folder, "full_tracks_" +
                                str(n_trks) + ".csv")
        df1.to_csv(df1_path, header=True, index=False)
        df_hs_path = os.path.join(folder, "hs_tracks_" +
                                  str(n_trks) + ".csv")
        df_hs.to_csv(df_hs_path, header=True, index=False)
        df_pu_path = os.path.join(folder, "pu_tracks_" +
                                  str(n_trks) + ".csv")
        df_pu.to_csv(df_pu_path, header=True, index=False)

def df_creator(df_temp1, lab_columns):
    """
    Dataframe creator for the full, HS and PU
    datasets.
    :param df_temp1: temp_df
    :param lab_columns: columns labels
    :return: full_df, HS_df, PU_df
    """
    df_full = df_temp1.loc[lambda dataset: df_temp1["linked_type"] < 2]
    df_full.drop(lab_columns, axis=1, inplace=True)
    hs_temp_df = df_temp1.loc[lambda dataset: df_temp1["linked_type"] == 0]
    pu_temp_df = df_temp1.loc[lambda dataset: df_temp1["linked_type"] == 1]
    hs_temp_df.drop(lab_columns, axis=1, inplace=True)
    pu_temp_df.drop(lab_columns, axis=1, inplace=True)
    return  df_full, hs_temp_df, pu_temp_df

