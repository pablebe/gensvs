
import glob
import os
import torch
import torch.nn.functional as F
import json
import copy
import numpy as np
import huggingface_hub as hf_hub

from tqdm import tqdm
from os import makedirs
from soundfile import write
from torchaudio import load
from os.path import join, dirname
from argparse import ArgumentParser
from librosa import resample
from sgmsvs.sgmse.util.other import set_torch_cuda_arch_list
set_torch_cuda_arch_list()
from bigvgan_utils.bigvgan import BigVGAN
from baseline_models.MSS_mask_model import MaskingModel
from bigvgan_utils.env import AttrDict
from bigvgan_utils.utils import load_checkpoint
from bigvgan_utils.meldataset import mel_spectrogram
from sgmsvs.loudness import calculate_loudness

SAVE_MELROFORM_AUDIO = False
FADE_LEN = 0.1 # seconds
LOUDNESS_LEVEL = -18 # dBFS

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the test data')
    parser.add_argument("--out_dir", type=str, required=True, help='Directory containing the enhanced data')
    parser.add_argument("--model_type", type=str, default="sgmsvs", choices=["sgmsvs", "melroformer_small_bigvgan"], help='Type of model to use for inference')
    parser.add_argument("--melroformer_ckpt", type=str,  help='Path to model checkpoint')
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference. Choose from 'cpu' or 'cuda:<device_id>' (e.g., 'cuda:0').")
    parser.add_argument("--output_mono", action="store_true", default=False, help="Whether to output mono audio.")
    parser.add_argument("--loudness_normalize", action="store_true", default=False, help="Whether to normalize the loudness of the output audio.")

    args = parser.parse_args()
    
    if args.device.startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[-1]
    elif args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        raise ValueError("Invalid device specified. Use 'cpu' or 'cuda:<device_id>'.")
    
    
    if args.model_type == "sgmsvs":
        os.makedirs('./trained_models', exist_ok=True)
        #check if model_checkpoints are downloaded into trained_models folder
        if not os.path.exists('./trained_models/sgmsvs/sgmsvs_epoch=510-sdr=7.22.ckpt'):
            hf_hub.download_file(repo_id="pablebe/sgmsvs", filename="sgmsvs.sgmsvs_epoch=510-sdr=7.22.ckpt", local_dir="./trained_models")
        
        
        
        inference_sgmsvs(args)
    elif args.model_type == "melroformer_small_bigvgan":
        os.makedirs('./trained_models', exist_ok=True)
        #check if model_checkpoints are downloaded into trained_models folder
        if not os.path.exists('./trained_models/sgmsvs/sgmsvs_epoch=510-sdr=7.22.ckpt'):
            hf_hub.download_file(repo_id="pablebe/sgmsvs", filename="sgmsvs.sgmsvs_epoch=510-sdr=7.22.ckpt", local_dir="./trained_models")
        # check if model_checkpoints are downloaded into trained_models folder
    
        parser.add_argument("--bigvgan_config_file", type=str, default=None, required=True, help="Path to the config file for the BigVGAN model.")
        parser.add_argument("--bigvgan_checkpoint", type=str, default=None, required=True, help="Path to the checkpoint file for the BigVGAN model.")
        parser.add_argument("--bigvgan_use_cuda_kernel", action="store_true", default=False, help="Whether to use the CUDA kernel for the BigVGAN model.")
    
    
        inference_melroformer_small_bigvgan(args)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}. Choose from 'sgmsvs' or 'melroformer_small_bigvgan'.")
        
