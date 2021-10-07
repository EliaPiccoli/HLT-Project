import pandas as pd
import numpy as np
import pickle

from data_parser import Parser
from torch.utils.data import Dataset

class BalancedCTTDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame, offices_path: str, seq_len=30, parser=None, task='FirstLevel', class_to_label_path="models/class_to_label", verbose=False):
        self.seq_len = seq_len
        # read dataset to later fetch object, off, label
        self.dataset = dataset
        # create dictionary to transform off in onehot
        self.offices = pd.read_csv(offices_path, header=None)[0].to_numpy()
        offices_onehot = np.eye(len(self.offices))
        self.office2onehot = {k: v for k, v in zip(self.offices, offices_onehot)}
        # instantiate Parser for later use
        if parser is None:
            self.parser = Parser()
            self.parser.create_vocab(self.dataset['Oggetto'].to_numpy())
        else:
            self.parser = parser
        self.task = task
        self.class_to_label_dict = {}
        with open(class_to_label_path, 'rb') as handle:
            self.class_to_label_dict = pickle.load(handle)
        self.class_to_label_dict = self.class_to_label_dict['class_dict']
        self.verbose = verbose

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # take out row with index idx
        dataset_sample = self.dataset.iloc[idx]
        # take object and parse/tokenize it (needed for sx model)
        obj = self.parser.encode_element(dataset_sample['Oggetto'], size=self.seq_len)
        # get onehot for particular off (needed for dx model)
        office = self.office2onehot[dataset_sample['Contenitore']]
        # simply fetch label (needed for final model)
        if self.task == 'FirstLevel':
            label, class_code = self.class_to_label_dict[(dataset_sample['PrimoLivello'], -1)]
        else:
            label, class_code = self.class_to_label_dict[(dataset_sample['PrimoLivello'], dataset_sample['SecondoLivello'])]
        if self.verbose:
            print("Index:", idx)
            print("Object:", dataset_sample['Oggetto'])
            print("Parsed Object:", self.parser.tokenize_string(self.parser.filter_string(dataset_sample['Oggetto'])))
            print("Office:", dataset_sample['Contenitore'])
            print("---------------")

        return obj, office, label, class_code