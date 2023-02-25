import os
import sys
import math
import pprint

import torch

from torchdrug import core, models, tasks, datasets, utils
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import util
from gearnet import model, layer


def save(solver, path):
    if isinstance(solver.model, tasks.Unsupervised):
        model = solver.model.model.model
    else:
        model = solver.model.model

    if comm.get_rank() == 0:
        logger.warning("Save checkpoint to %s" % path)
    path = os.path.expanduser(path)
    if comm.get_rank() == 0:
        torch.save(model.state_dict(), path)
    comm.synchronize()


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    species_start = cfg.dataset.get("species_start", 0)
    species_end = cfg.dataset.get("species_end", 0)
    assert species_end >= species_start
    if species_end > species_start:
        cfg.dataset.species_id = species_start
        cfg.dataset.split_id = 0
        cfg.dataset.pop("species_start")
        cfg.dataset.pop("species_end")
    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_pretrain_solver(cfg, dataset)

    step = cfg.get("save_interval", 1)
    for i in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
        
        if species_end == species_start:
            solver.train(**kwargs)
        else:
            for species_id in range(species_start, species_end):
                for split_id in range(dataset.species_nsplit[species_id]):
                    cfg.dataset.species_id = species_id
                    cfg.dataset.split_id = split_id
                    dataset = core.Configurable.load_config_dict(cfg.dataset)
                    logger.warning('Epoch: {}\tSpecies id: {}\tSplit id: {}\tSplit length: {}'.format(
                                i, species_id, split_id, len(dataset)))
                    solver.train_set = dataset
                    solver.train(**kwargs)

        save(solver, "model_epoch_%d.pth" % (i + kwargs["num_epoch"]))
