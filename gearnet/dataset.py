
import os
import glob
import h5py
import torch
import warnings

from tqdm import tqdm

from torch.utils import data as torch_data

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.Fold3D")
class Fold3D(data.ProteinDataset):

    url = "https://zenodo.org/record/7593591/files/fold3d.zip"
    md5 = "7b052a94afa4c66f9bebeb9efd769186"
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
                split_path = utils.extract(os.path.join(path, "%s.zip" % split))
                pdb_files += sorted(glob.glob(os.path.join(split_path, split, "*.hdf5")))
            self.load_hdf5s(pdb_files, verbose=verbose, **kwargs)
            self.save_pickle(pkl_file, verbose=verbose)

        label_files = [os.path.join(path, '%s.txt' % split) for split in self.splits]
        class_map = os.path.join(path, 'class_map.txt')
        label_list = self.get_label_list(label_files, class_map)
        fold_labels = [label_list[os.path.basename(pdb_file)[:-5]] for pdb_file in self.pdb_files]
        self.targets = {'fold_label': fold_labels}

        splits = [os.path.basename(os.path.dirname(pdb_file)) for pdb_file in self.pdb_files]
        self.num_samples = [splits.count(split) for split in self.splits]

    def load_hdf5(self, hdf5_file):
        h5File = h5py.File(hdf5_file, "r")
        node_position = torch.as_tensor(h5File["atom_pos"][(0)])
        num_atom = node_position.shape[0]
        atom_type = torch.as_tensor(h5File["atom_types"][()])
        atom_name = h5File["atom_names"][()]
        atom_name = torch.as_tensor([data.Protein.atom_name2id.get(name.decode(), -1) for name in atom_name])
        atom2residue = torch.as_tensor(h5File["atom_residue_id"][()])
        residue_type_name = h5File["atom_residue_names"][()]
        residue_type = []
        residue_feature = []
        lst_residue = -1
        for i in range(num_atom):
            if atom2residue[i] != lst_residue:
                residue_type.append(data.Protein.residue2id.get(residue_type_name[i].decode(), 0))
                residue_feature.append(data.feature.onehot(residue_type_name[i].decode(), data.feature.residue_vocab, allow_unknown=True))
                lst_residue = atom2residue[i]
        residue_type = torch.as_tensor(residue_type)
        residue_feature = torch.as_tensor(residue_feature)
        num_residue = residue_type.shape[0]
       
        '''
        edge_list = torch.cat([
            torch.as_tensor(h5File["cov_bond_list"][()]),
            torch.as_tensor(h5File["cov_bond_list_hb"][()])
        ], dim=0)
        bond_type = torch.zeros(edge_list.shape[0], dtype=torch.long)
        edge_list = torch.cat([edge_list, bond_type.unsqueeze(-1)], dim=-1)
        '''
        edge_list = torch.as_tensor([[0, 0, 0]])
        bond_type = torch.as_tensor([0])

        protein = data.Protein(edge_list, atom_type, bond_type, num_node=num_atom, num_residue=num_residue,
                               node_position=node_position, atom_name=atom_name,
                                atom2residue=atom2residue, residue_feature=residue_feature, 
                                residue_type=residue_type)
        return protein

    def load_hdf5s(self, hdf5_files, transform=None, lazy=False, verbose=0):
        num_sample = len(hdf5_files)
        if num_sample > 1000000:
            warnings.warn("Preprocessing proteins of a large dataset consumes a lot of CPU memory and time. "
                          "Use load_pdbs(lazy=True) to construct molecules in the dataloader instead.")

        self.transform = transform
        self.lazy = lazy
        self.data = []
        self.pdb_files = []
        self.sequences = []

        if verbose:
            hdf5_files = tqdm(hdf5_files, "Constructing proteins from pdbs")
        for i, hdf5_file in enumerate(hdf5_files):
            if not lazy or i == 0:
                protein = self.load_hdf5(hdf5_file)
            else:
                protein = None
            if hasattr(protein, "residue_feature"):
                with protein.residue():
                    protein.residue_feature = protein.residue_feature.to_sparse()
            self.data.append(protein)
            self.pdb_files.append(hdf5_file)
            self.sequences.append(protein.to_sequence() if protein else None)

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
        if self.lazy:
            protein = self.load_hdf5(self.pdb_files[index])
        else:
            protein = self.data[index].clone()
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        item = {"graph": protein, "fold_label": self.targets["fold_label"][index]}
        if self.transform:
            item = self.transform(item)
        return item
