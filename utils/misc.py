import logging
import numpy
import uproot
import json


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
    f = uproot.open(rootfile_path)
    tree = f['RefitAnalysis;1']
    dataset = tree.pandas.df(df_columns)
    utils_log.info("before : {}".format(dataset.shape))
    features = dataset.loc[:, dataset.columns].values
    dataset = dataset.replace([numpy.inf, -numpy.inf], numpy.nan).dropna(how='any')
    dataset = dataset.drop_duplicates()
    utils_log.info("after checking nan and inf : {}".format(dataset.shape))
    return features, dataset


def print_df_info(df):
    n_merge_max = df["v_nMergeEvent"].max()
    n_merge_mean = df["v_nMergeEvent"].mean()
    n_evt_max = df["v_nEvent"].max()
    utils_log.info("############### USEFUL DATASET INFOS #####################")
    utils_log.info("dataset columns: {}".format(list(df.columns)))
    utils_log.info("dataset shape: {}".format(df.shape))
    utils_log.info("nEvt: {}".format(n_evt_max))
    utils_log.info("MAX Number of merged vertices per event: {}".format(n_merge_max))
    utils_log.info("AVERAGE Number of merged vertices per event: {}".format(n_merge_mean))
    utils_log.info("##########################################################")


