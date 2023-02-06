
import os
import glob
import torch

from torch.utils import data as torch_data

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.Fold3D")
class Fold3D(data.ProteinDataset):
    """
    Fold labels for a set of proteins determined by the global structural topology.

    Statistics:
        - #Train: 12,312
        - #Valid: 736
        - #Test: 718

    Parameters:
        path (str): the path to store the dataset
        test_split (str, optional): the test split used for evaluation
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "https://oxer11.github.io/fold3d.zip"
    md5 = "abd65e0265f6534d61ae3c8a6d23bb35"
    processed_file = "fold3d.pkl.gz"
    splits = ["train", "valid", "test_fold", "test_family", "test_superfamily"]

    def __init__(self, path, test_split="test_fold", verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        if test_split not in self.splits[-3:]:
            raise ValueError("Unknown test split `%s` for Fold3D dataset" % test_split)
        self.test_split = test_split

        zip_file = utils.download(self.url, path, md5=self.md5)
        path = os.path.join(utils.extract(zip_file), "fold3d")
        pkl_file = os.path.join(path, self.processed_file)

        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        else:
            pdb_files = []
            for split in self.splits:
                split_path = utils.extract(os.path.join(path, "%s_pdb.zip" % split))
                pdb_files += sorted(glob.glob(os.path.join(split_path, split, "*.pdb")))
            self.load_pdbs(pdb_files, verbose=verbose, **kwargs)
            self.save_pickle(pkl_file, verbose=verbose)

        label_files = [os.path.join(path, '%s.txt' % split) for split in self.splits]
        class_map = os.path.join(path, 'class_map.txt')
        label_list = self.get_label_list(label_files, class_map)
        fold_labels = [label_list[os.path.basename(pdb_file)[:-4]] for pdb_file in self.pdb_files]
        self.targets = {'fold_label': fold_labels}

        splits = [os.path.basename(os.path.dirname(pdb_file)) for pdb_file in self.pdb_files]
        self.num_samples = [splits.count(split) for split in self.splits]

    def get_label_list(self, label_files, classmap):
        with open(classmap, "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            class_map = dict([line.split('\t') for line in lines])
        label_list = {}
        for fname in label_files:
            label_file = open(fname, 'r')
            for line in label_file.readlines():
                line = line.strip().split('\t')
                name, label = line[0], line[-1]
                label_list[name] = torch.tensor(int(class_map[label])).long()
        return label_list

    def split(self):
        keys = ["train", "valid", self.test_split]
        offset = 0
        splits = []
        for split_name, num_sample in zip(self.splits, self.num_samples):
            if split_name in keys:
                split = torch_data.Subset(self, range(offset, offset + num_sample))
                splits.append(split)
            offset += num_sample
        return splits
    
    def get_item(self, index):
        protein = self.data[index].clone()
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        item = {"graph": protein, "fold_label": self.targets["fold_label"][index]}
        if self.transform:
            item = self.transform(item)
        return item
