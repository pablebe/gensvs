import os
import torch
import copy
import json
import glob
import numpy as np
import soundfile as sf
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from torchaudio import load
from librosa import resample

from bigvgan_utils.bigvgan import BigVGAN
from baseline_models.MSS_mask_model import MaskingModel
from bigvgan_utils.utils import load_checkpoint
from bigvgan_utils.env import AttrDict
from bigvgan_utils.meldataset import mel_spectrogram
from sgmsvs.sgmse.util.other import set_torch_cuda_arch_list
set_torch_cuda_arch_list()
from sgmsvs.MSS_model import ScoreModel
from sgmsvs.sgmse.util.other import pad_spec
from sgmsvs.loudness import calculate_loudness

#TODO: model inference is different ==> signals do not match to zenodo why?
#TODO: Get CUDA random seed from enhancement script on Lucier to make sgmsvs inference reproducible => also corrector step 1 was used for sgmsvs samples
TARGET_SR = 44100
T_EPS = 0.03
PAD_MODE = 'reflection'

#TODO: Check where seed need to be set!

class MelRoFoBigVGAN():
    def __init__(self, device='cuda'):
        os.makedirs('./trained_models/melroformer_small', exist_ok=True)
        os.makedirs('./trained_models/bigvgan_finetuned', exist_ok=True)

        melroformer_ckpt = os.path.join('trained_models', 'melroformer_small', 'melroformer_small_epoch=548-sdr=8.85.ckpt')
        bigvgan_checkpoint = os.path.join('trained_models', 'bigvgan_finetuned', 'g_05570000.ckpt')
        
        if not os.path.exists(melroformer_ckpt):
            hf_hub_download(repo_id="pablebe/melroformer_small", filename="melroformer_small_epoch=548-sdr=8.85.ckpt", local_dir=os.path.join("trained_models", "melroformer_small"))

        if not os.path.exists(bigvgan_checkpoint):
            hf_hub_download(repo_id="pablebe/bigvgan_finetuned", filename="g_05570000.ckpt", local_dir=os.path.join("trained_models", "bigvgan_finetuned"))

        self.device = device
        melroform_bigvgan_model = MaskingModel.load_from_checkpoint(melroformer_ckpt, map_location=device)
        # clone melrformer model
        self.melroformer = copy.deepcopy(melroform_bigvgan_model.dnn)
        
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'bigvgan_utils', 'configs', 'bigvgan_v2_44khz_128band_512x.json')) as f:
            data = f.read()
        json_config = json.loads(data)
        self.bigvgan_config = AttrDict(json_config)
        self.bigvgan = BigVGAN(self.bigvgan_config).to(device)
        state_dict_g = load_checkpoint(bigvgan_checkpoint, device)
        self.bigvgan.load_state_dict(state_dict_g["generator"])
        self.bigvgan.eval()

        
    def forward(self, y):
        num_frames = int(np.ceil(y.shape[1] / self.bigvgan_config.hop_size))
        target_len = (num_frames ) * self.bigvgan_config.hop_size
        current_len = y.size(-1)
        pad = max(target_len - current_len, 0)
        if pad != 0:
            y = F.pad(y, (pad//2, pad//2+(pad%2)), mode='constant')
        
        norm_fac = y.abs().max()
        y = y / norm_fac

        x_hat_melrofo = self.melroformer(y.unsqueeze(0).to(self.device)).squeeze()
        x_hat_melrofo = x_hat_melrofo.to("cpu") * norm_fac
        mel_sep = mel_spectrogram(
                                    x_hat_melrofo.squeeze(),
                                    self.bigvgan_config.n_fft,
                                    self.bigvgan_config.num_mels,
                                    self.bigvgan_config.sampling_rate,
                                    self.bigvgan_config.hop_size,
                                    self.bigvgan_config.win_size,
                                    self.bigvgan_config.fmin,
                                    self.bigvgan_config.fmax_for_loss,
                                )
                
        mel_sep = mel_sep.to(self.device)
        x_hat = self.bigvgan(mel_sep).squeeze()
        x_hat = x_hat[:,pad//2:-(pad//2+(pad%2))]
        x_hat_melrofo = x_hat_melrofo[:,pad//2:-(pad//2+(pad%2))]
        
        return x_hat, x_hat_melrofo
    
    def run_folder(self, 
                   test_dir,
                   out_dir,
                   loudness_normalize=False, 
                   loudness_level=-18, 
                   output_mono=False):
        
        os.makedirs(out_dir, exist_ok=True)
        noisy_files = []
        if 'musdb18hq' in test_dir:
            noisy_files += sorted(glob.glob(os.path.join(test_dir, 'mixture.wav')))
            noisy_files += sorted(glob.glob(os.path.join(test_dir, '**', 'mixture.wav')))
        else:
            noisy_files += sorted(glob.glob(os.path.join(test_dir, '*.wav')))
            noisy_files += sorted(glob.glob(os.path.join(test_dir, '**', '*.wav')))

        # Enhance files
        for noisy_file in tqdm(noisy_files, desc="Processing files with MelRoFo(S)+BigVGAN"):
            filename = noisy_file.replace(test_dir, "")
            filename = filename[1:] if filename.startswith(os.path.sep) else filename

            # load wav
            y, sr = load(noisy_file)
            # Resample if necessary
            if sr != TARGET_SR:
                y = torch.tensor(resample(y.numpy(), orig_sr=sr, target_sr=TARGET_SR))
            
            if y.shape[0]<2:
                # if audio has only one channel copy and stack channel to get stereo input for model 
                y = torch.stack((y, y), dim=0).squeeze()
                            
            with torch.no_grad():
                x_hat, x_hat_melrofo = self.forward(y)
            
            if y.shape[0]>1:
                #if stereo put channel dimenion last
                x_hat = x_hat.T
                x_hat_melrofo = x_hat_melrofo.T
                
            if output_mono:           
                x_hat = x_hat[:,0].cpu()
#                x_hat = np.stack((audio_mono, audio_mono), axis=1)
                
                x_hat_melrofo = x_hat_melrofo[:,0].cpu()
#                x_hat_melrofo = np.stack((audio_mono_melrofo, audio_mono_melrofo), axis=1)
            else:
                x_hat = x_hat.cpu().numpy()
                x_hat_melrofo = x_hat_melrofo.cpu().numpy()
                
            if loudness_normalize:
                # Normalize loudness
                L_audio = calculate_loudness(x_hat, sr)
                L_diff_goal_audio = loudness_level - L_audio
                k_scale_audio = 10**(L_diff_goal_audio/20)
                x_hat = x_hat * k_scale_audio
                
                L_audio_melrofo = calculate_loudness(x_hat_melrofo, sr)
                L_diff_goal_audio_melrofo = loudness_level - L_audio_melrofo
                k_scale_audio = 10**(L_diff_goal_audio_melrofo/20)
                x_hat_melrofo = x_hat_melrofo * k_scale_audio
            
            # Write separated vocals from MelRoFo(S)+BigVGAN to wav file
            filename = 'separated_vocals_'+filename
            os.makedirs(os.path.dirname(os.path.join(out_dir, 'melroformer_bigvgan',filename)), exist_ok=True)
            sf.write(os.path.join(out_dir, 'melroformer_bigvgan', filename), x_hat, TARGET_SR)
            # Write separated vocals from MelRoFo(S) to wav file
            os.makedirs(os.path.dirname(os.path.join(out_dir,'melroformer_small',filename)), exist_ok=True)
            sf.write(os.path.join(out_dir,'melroformer_small', filename), x_hat_melrofo, TARGET_SR)
                
                
class SGMSVS():
    def __init__(self, device='cuda'):
        os.makedirs('./trained_models/sgmsvs', exist_ok=True)

        sgmsvs_ckpt = os.path.join('trained_models', 'sgmsvs', 'sgmsvs_epoch=510-sdr=7.22.ckpt')

        if not os.path.exists(sgmsvs_ckpt):
            hf_hub_download(repo_id="pablebe/sgmsvs", filename="sgmsvs_epoch=510-sdr=7.22.ckpt", local_dir=os.path.join("trained_models", "sgmsvs"))

        self.device = device
        self.model = ScoreModel.load_from_checkpoint(sgmsvs_ckpt, map_location=self.device)
        self.model.t_eps = T_EPS
        self.model.eval()
        
    def forward(self, 
                y, 
                sampler_type='pc', 
                corrector='ald',
                corrector_steps=2,
                N=45,
                snr=0.5, 
                output_mono=False,
                ch_by_ch_processing=False,
                random_seed=1234
               ):


        torch.manual_seed(random_seed)

        T_orig = y.size(1)

        # Normalize
        norm_factor = y.abs().max()
        y = y / norm_factor
        
        with torch.no_grad():
            # Prepare DNN input
            if y.shape[0]>1:
                Y = torch.unsqueeze(self.model._forward_transform(self.model._stft(y.to(self.device))), 1)
            else:
                Y = torch.unsqueeze(self.model._forward_transform(self.model._stft(y.to(self.device))), 0)
            Y = pad_spec(Y, mode=PAD_MODE)

            x_hat_ch = []
            if ch_by_ch_processing:
                for ch in range(Y.shape[0]):
                    if self.model.sde.__class__.__name__ == 'OUVESDE':
                        if sampler_type == 'pc':
                            sampler = self.model.get_pc_sampler('reverse_diffusion', corrector, Y[ch,...][None,...].to(self.device), N=N, 
                                corrector_steps=corrector_steps, snr=snr)
                        elif sampler_type == 'ode':
                            sampler = self.model.get_ode_sampler(Y[ch,...][None,...].to(self.device), N=N)
                        else:
                            raise ValueError(f"Sampler type {sampler_type} not supported")
                    elif self.model.sde.__class__.__name__ == 'SBVESDE':
                        sampler_type = 'ode' if sampler_type == 'pc' else sampler_type
                        sampler = self.model.get_sb_sampler(sde=self.model.sde, y=Y[ch,...][None,...].cuda(), sampler_type=sampler_type)
                    else:
                        raise ValueError(f"SDE {self.model.sde.__class__.__name__} not supported")
                    sample, _ = sampler()
                    
                    # Backward transform in time domain
                    x_hat = self.model.to_audio(sample.squeeze(), T_orig)
                    x_hat_ch.append(x_hat)
                x_hat = torch.stack(x_hat_ch, dim=0)
            else:
                if output_mono:
                    if self.model.sde.__class__.__name__ == 'OUVESDE':
                        if sampler_type == 'pc':
                            sampler = self.model.get_pc_sampler('reverse_diffusion', corrector, Y[0,...][None,...].to(self.device), N=N, 
                                corrector_steps=corrector_steps, snr=snr)
                        elif sampler_type == 'ode':
                            sampler = self.model.get_ode_sampler(Y[0,...][None,...].to(self.device), N=N)
                        else:
                            raise ValueError(f"Sampler type {sampler_type} not supported")
                    elif self.model.sde.__class__.__name__ == 'SBVESDE':
                        sampler_type = 'ode' if sampler_type == 'pc' else sampler_type
                        sampler = self.model.get_sb_sampler(sde=self.model.sde, y=Y[0,...][None,...].cuda(), sampler_type=sampler_type)
                    else:
                        raise ValueError(f"SDE {self.model.sde.__class__.__name__} not supported")
                    sample, _ = sampler()
                    
                    # Backward transform in time domain
                    x_hat = self.model.to_audio(sample.squeeze(), T_orig)                
                else:
                    if self.model.sde.__class__.__name__ == 'OUVESDE':
                        if sampler_type == 'pc':
                            sampler = self.model.get_pc_sampler('reverse_diffusion', corrector, Y.to(self.device), N=N, 
                                corrector_steps=corrector_steps, snr=snr)
                        elif sampler_type == 'ode':
                            sampler = self.model.get_ode_sampler(Y.to(self.device), N=N)
                        else:
                            raise ValueError(f"Sampler type {sampler_type} not supported")
                    elif self.model.sde.__class__.__name__ == 'SBVESDE':
                        sampler_type = 'ode' if sampler_type == 'pc' else sampler_type
                        sampler = self.model.get_sb_sampler(sde=self.model.sde, y=Y.cuda(), sampler_type=sampler_type)
                    else:
                        raise ValueError(f"SDE {self.model.sde.__class__.__name__} not supported")
                    sample, _ = sampler()
                    
                    # Backward transform in time domain
                    x_hat = self.model.to_audio(sample.squeeze(), T_orig)
        # Renormalize
        x_hat = x_hat * norm_factor

        if output_mono:
            x_hat = x_hat.unsqueeze(0)

        return x_hat

    def run_folder(self, 
                   test_dir, 
                   out_dir, 
                   sampler_type='pc', 
                   corrector='ald', 
                   corrector_steps=2, 
                   N=45, 
                   snr=0.5,
                   random_seed=1234,
                   loudness_normalize=False, 
                   loudness_level=-18, 
                   output_mono=False,
                   ch_by_ch_processing=False):
        
        os.makedirs(out_dir, exist_ok=True)
        noisy_files = []
        if 'musdb18hq' in test_dir:
            noisy_files += sorted(glob.glob(os.path.join(test_dir, 'mixture.wav')))
            noisy_files += sorted(glob.glob(os.path.join(test_dir, '**', 'mixture.wav')))
        else:
            noisy_files += sorted(glob.glob(os.path.join(test_dir, '*.wav')))
            noisy_files += sorted(glob.glob(os.path.join(test_dir, '**', '*.wav')))
        
        for noisy_file in tqdm(noisy_files, desc="Processing files with SGMSVS"):
            filename = noisy_file.replace(test_dir, "")
            filename = filename[1:] if filename.startswith(os.path.sep) else filename
            
            y, sr = load(noisy_file)

            # Resample if necessary
            if sr != TARGET_SR:
                y = torch.tensor(resample(y.numpy(), orig_sr=sr, target_sr=TARGET_SR))
            

            #if output_mono:
            # if audio is written as mono output only process first channel => sgmsvs is faster for mono input
            #    x_hat = self.forward(y[0,:].unsqueeze(0), sampler_type=sampler_type, corrector=corrector, corrector_steps=corrector_steps, N=N, snr=snr, output_mono=output_mono)
            #else:
            x_hat = self.forward(y, sampler_type=sampler_type, corrector=corrector, corrector_steps=corrector_steps, N=N, snr=snr, output_mono=output_mono, ch_by_ch_processing=ch_by_ch_processing, random_seed=random_seed)

            x_hat = x_hat.T
                
            if output_mono:           
                x_hat = x_hat[:,0].cpu()
#                x_hat = np.stack((audio_mono, audio_mono), axis=1)
            else:
                x_hat = x_hat.cpu().numpy()
                
            if loudness_normalize:
                # Normalize loudness
                L_audio = calculate_loudness(x_hat, sr)
                L_diff_goal_audio = loudness_level - L_audio
                k_scale_audio = 10**(L_diff_goal_audio/20)
                x_hat = x_hat * k_scale_audio

            
            # Write separated vocals to wav file
            filename = 'separated_vocals_'+filename
            os.makedirs(os.path.dirname(os.path.join(out_dir, 'sgmsvs',filename)), exist_ok=True)
            sf.write(os.path.join(out_dir, 'sgmsvs', filename), x_hat, TARGET_SR)

