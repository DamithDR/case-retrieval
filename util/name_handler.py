from data.coliee import coliee
from data.ecthr import ecthr
from data.ilpcr import ilpcr
from data.irled import irled
from data.muser import muser
from models.flagembed import flag
from models.nvembedv2 import Nvembedv2
from models.sftembed import sfr
from models.stellaembed import stella


def get_data_class(dataset):
    if dataset == 'ilpcr':
        return ilpcr()
    elif dataset == 'coliee':
        return coliee()
    elif dataset == 'irled':
        return irled()
    elif dataset == 'muser':
        return muser()
    elif dataset == 'ecthr':
        return ecthr()


def get_model_class(model_name):
    if model_name == 'nvidia/NV-Embed-v2':
        return Nvembedv2()
    elif model_name == 'BAAI/bge-en-icl':
        return flag()
    elif model_name == 'Salesforce/SFR-Embedding-2_R':
        return sfr()
    elif model_name == 'dunzhang/stella_en_1.5B_v5':
        return stella()


def standadise_name(name):
    return name.split('/')[-1] if name.__contains__('/') else name


def get_save_names(model_class, data_class):
    model_name = model_class.get_name()
    data_name = data_class.get_name()
    return standadise_name(model_name), standadise_name(data_name)
