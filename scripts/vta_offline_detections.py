import torch, pytorch_lightning as pl, tempfile, os
from mot_neural_solver.path_cfg import OUTPUT_PATH, DATA_PATH
import os.path as osp

from mot_neural_solver.pl_module.pl_module import MOTNeuralSolver

import pandas as pd

from sacred import SETTINGS

import traceback
import yaml

import time
from os import path as osp

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.datasets.factory import Datasets
from tracktor.tracker import Tracker

from mot_neural_solver.path_cfg import OUTPUT_PATH

from mot_neural_solver.data.seq_processing.MOTCha_loader import MOV_CAMERA_DICT as MOT17_MOV_CAMERA_DICT
from mot_neural_solver.data.seq_processing.MOT15_loader import MOV_CAMERA_DICT as MOT15_MOV_CAMERA_DICT
from mot_neural_solver.data.preprocessing import FRCNNPreprocessor


SETTINGS.CONFIG.READ_ONLY_CONFIG=False

def patch_checkpoint_file(ckpt_file_path: str) -> str:
    """
    Adds missing metadata to .ckpt file to ensure compatibility with newer pytorch versions
    """
    checkpoint = torch.load(ckpt_file_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "pytorch-lightning_version" in checkpoint and "state_dict" in checkpoint:
        return ckpt_file_path

    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        checkpoint = {"state_dict": checkpoint}

    # add missing metadata
    checkpoint.setdefault("pytorch-lightning_version", pl.__version__)
    checkpoint.setdefault("hyper_parameters", {})

    fd, updated_checkpoint_path = tempfile.mkstemp(suffix=".ckpt")
    os.close(fd)
    torch.save(checkpoint, updated_checkpoint_path)
    return updated_checkpoint_path

def process_detections(config, output_folder_path):
    ckpt_path = config['ckpt_path'] if osp.exists(config['ckpt_path']) else osp.join(OUTPUT_PATH, config['ckpt_path'])

    ckpt_path = patch_checkpoint_file(ckpt_path)

    model = MOTNeuralSolver.load_from_checkpoint(checkpoint_path=ckpt_path, hparams=config)

    model.hparams.update({'eval_params':config['eval_params'],
                          'data_splits':config['data_splits']})
    model.hparams['dataset_params']['precomputed_embeddings'] = config['dataset_params']['precomputed_embeddings']
    model.hparams['dataset_params']['img_batch_size'] = config['dataset_params']['img_batch_size']

    # Get output MOT results files
    test_dataset = model.test_dataset()
    model.track_all_seqs(dataset=test_dataset,
                                              output_files_dir = output_folder_path,
                                              use_gt = False,
                                              verbose=True)

    print("Finished processing")

def preprocess_detection(config):
    if config['prepr_w_tracktor']:
        prepr_params = config['tracktor_params']

    else:
        prepr_params = config['frcnn_prepr_params']

    MOV_CAMERA_DICT = {**MOT15_MOV_CAMERA_DICT, **MOT17_MOV_CAMERA_DICT, 'det':False}

    # object detection
    print("Initializing object detector.")
    obj_detect = FRCNN_FPN(num_classes=2)
    obj_detect.load_state_dict(torch.load(osp.join(OUTPUT_PATH, config['frcnn_weights']),
                                          map_location=lambda storage, loc: storage))
    obj_detect.eval()
    obj_detect.cuda()

    if config['prepr_w_tracktor']:
        preprocessor = Tracker(obj_detect, None, prepr_params['tracker'])
    else:
        preprocessor = FRCNNPreprocessor(obj_detect, prepr_params)

    print(f"Starting  preprocessing of datasets {config['dataset_names']} with {'Tracktor' if config['prepr_w_tracktor'] else 'FRCNN'} \n")

    for dataset_name in config['dataset_names']:
        dataset = Datasets(dataset_name)
        print(f"Preprocessing {len(dataset)} sequences from dataset {dataset_name} \n")

        time_total = 0
        num_frames = 0
        for seq in dataset:
            preprocessor.reset()

            start = time.time()
            print(f"Preprocessing : {seq}")
            if config['prepr_w_tracktor']:
                preprocessor.do_align = config['tracktor_params']['tracker']['do_align'] and (MOV_CAMERA_DICT[str(seq)])

            data_loader = DataLoader(seq, batch_size=1, shuffle=False, num_workers=6, pin_memory=True)
            for i, frame in enumerate(tqdm(data_loader)):
                with torch.no_grad():
                    preprocessor.step(frame)
                num_frames += 1

            time_total += time.time() - start
            print(f"Runtime for {seq}: {time.time() - start :.1f} s.")

            output_file_path = osp.join(seq.seq_path, 'det', prepr_params['det_file_name'])
            if config['prepr_w_tracktor']:
                results = preprocessor.get_results()
                seq.write_results(results, output_file_path)
            else:
                print(f"Writing predictions in: {output_file_path}")
                preprocessor.save_results(output_file_path)

        print(f"Tracking runtime for all sequences (without evaluation or image writing): "
                  f"{time_total:.1f} s ({num_frames / time_total:.1f} Hz)")


def process_detections_with_MPNTrack(config, output_folder_path):
    preprocess_detection(config.preprocessing)
    print("Succesfully preprocessed detection file\n\n")
    process_detections(config, output_folder_path)

    