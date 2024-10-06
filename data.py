import json
import os
import pickle


def read_json(path, as_int=False):
    with open(path, 'r') as f:
        raw = json.load(f)
        if as_int:
            data = dict((int(key), value) for (key, value) in raw.items())
        else:
            data = dict((key, value) for (key, value) in raw.items())
        del raw
        return data

def load_dataset(args):
    train = read_json(os.path.join(args.data_path, args.train_file), True)
    val = read_json(os.path.join(args.data_path, args.dev_file), True)
    test = read_json(os.path.join(args.data_path, args.test_file), True)
    item_meta_dict = json.load(open(os.path.join(args.data_path, args.meta_file)))

    item2id = read_json(os.path.join(args.data_path, args.item2id_file))
    id2item = {v: k for k, v in item2id.items()}

    item_meta_dict_filted = dict()
    for k, v in item_meta_dict.items():
        if k in item2id:
            item_meta_dict_filted[k] = v

    return train, val, test, item_meta_dict_filted, item2id, id2item