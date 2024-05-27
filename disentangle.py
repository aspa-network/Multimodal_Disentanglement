import gc
import logging
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from config import get_config_regression
from multimodal_data_loader import MMDataLoader
from trains import ATIO
from utils import setup_seed
from trains.singleTask.model import dns
from trains.singleTask.distillnets import get_distillation_kernel, get_distillation_kernel_homo
from trains.singleTask.misc import softmax
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
logger = logging.getLogger('MMSA')

def _set_logger(log_dir, model_name, dataset_name, verbose_level):

    # base logger
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}.log"
    logger = logging.getLogger('MMSA')
    logger.setLevel(logging.DEBUG)

    # file handler
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # stream handler
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


def Disentanglement_NS(
    model_name, dataset_name, config=None, config_file="", seeds=[], is_tune=False,
    model_save_dir="", res_save_dir="", log_dir="",
    num_workers=4, verbose_level=1, mode = '', is_distill = False
):
    # Initialization
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()
    
    seeds = seeds if seeds != [] else [1111, 1112, 1113, 1114, 1115]
    logger = _set_logger(log_dir, model_name, dataset_name, verbose_level)
    

    args = get_config_regression(model_name, dataset_name, config_file)
    args.is_distill = is_distill  # use or not use distill, train use, test not use
    args.mode = mode # train or test
    args['model_save_path'] = Path(model_save_dir) / f"{args['model_name']}-{args['dataset_name']}.pth"

    if config:
        args.update(config)


    res_save_dir = Path(res_save_dir) / "normal"
    res_save_dir.mkdir(parents=True, exist_ok=True)
    model_results = []
    for i, seed in enumerate(seeds):
        setup_seed(seed)
        args['cur_seed'] = i + 1
        result = _run(args, num_workers, is_tune)
        model_results.append(result)
    if args.is_distill:
        criterions = list(model_results[0].keys())
        # save result to csv
        csv_file = res_save_dir / f"{dataset_name}.csv"
        if csv_file.is_file():
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame(columns=["Model"] + criterions)
        # save results
        res = [model_name]
        for c in criterions:
            values = [r[c] for r in model_results]
            mean = round(np.mean(values)*100, 2)
            std = round(np.std(values)*100, 2)
            res.append((mean, std))
        df.loc[len(df)] = res
        df.to_csv(csv_file, index=None)
        logger.info(f"Results saved to {csv_file}.")


def _run(args, num_workers=4, is_tune=False, from_sena=False):

    dataloader = MMDataLoader(args, num_workers)
    if args.is_distill:
        print("training for DMD")

        # param of homogeneous graph distillation
        args.gd_size_low = 64  # hidden size of graph distillation
        args.w_losses_low = [1, 10]  # weights for losses: [logit, repr]
        args.metric_low = 'l1'  # distance metric for distillation loss

        # param of heterogeneous graph distillation
        args.gd_size_high = 32  # hidden size of graph distillation
        args.w_losses_high = [1, 10]  # weights for losses: [logit, repr]
        args.metric_high = 'l1'  # distance metric for distillation loss

        from_idx = [0, 1, 2]  # all modalities can be distilled from each other simultaneously
        assert len(from_idx) >= 1

        model = []
        model_dmd = getattr(dns, 'DMD')(args)
        model_distill_homo = getattr(get_distillation_kernel_homo, 'DistillationKernel')(hyp_params=args)

        model_distill_hetero = getattr(get_distillation_kernel, 'DistillationKernel')(hyp_params=args)

        model_dmd, model_distill_homo, model_distill_hetero = model_dmd.cuda(), model_distill_homo.cuda(), model_distill_hetero.cuda()

        model = [model_dmd, model_distill_homo, model_distill_hetero]
    else:
        print("testing phase for DNS")
        model = getattr(dns, 'DNS')(args)
        model = model.cuda()

    trainer = ATIO().getTrain(args)


    if args.mode == 'test':
        model.load_state_dict(torch.load('pt/mosi-aligned.pth'))
        results = trainer.do_test(model, dataloader['test'], mode="TEST")
        sys.stdout.flush()
        input('[Press Any Key to start another run]')
    else:
        epoch_results = trainer.do_train(model, dataloader, return_epoch_results=from_sena)
        model[0].load_state_dict(torch.load('pt/DNS.pth'))

        results = trainer.do_test(model[0], dataloader['test'], mode="TEST")

        del model
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)
    return results