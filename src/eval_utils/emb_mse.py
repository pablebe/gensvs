import tempfile
import torch
import numpy as np
import torchaudio
import subprocess
import os
import shutil
import traceback
from typing import Union
from pathlib import Path
from hypy_utils import write
from hypy_utils.logging_utils import setup_logger
from hypy_utils.tqdm_utils import tmap
from .model_loader import ModelLoader
from .utils import *

TORCHAUDIO_RESAMPLING = True
PathLike = Union[str, Path]
log = setup_logger()
sox_path = os.environ.get('SOX_PATH', 'sox')
ffmpeg_path = os.environ.get('FFMPEG_PATH', 'ffmpeg')

if not(TORCHAUDIO_RESAMPLING):
    if not shutil.which(sox_path):
        log.error(f"Could not find SoX executable at {sox_path}, please install SoX and set the SOX_PATH environment variable.")
        exit(3)
    if not shutil.which(ffmpeg_path):
        log.error(f"Could not find ffmpeg executable at {ffmpeg_path}, please install ffmpeg and set the FFMPEG_PATH environment variable.")
        exit(3)


class EmbeddingMSE:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    loaded = False

    def __init__(self, ml: ModelLoader, audio_load_worker=8, load_model=True):
        self.ml = ml
        self.audio_load_worker = audio_load_worker
        self.sox_formats = find_sox_formats(sox_path)

        if load_model:
            self.ml.load_model()
            self.loaded = True

        # Disable gradient calculation because we're not training
        torch.autograd.set_grad_enabled(False)

    def load_audio(self, f: Union[str, Path]):
        f = Path(f)

        # Create a directory for storing normalized audio files
        cache_dir = f.parent / "convert" / str(self.ml.sr)
        new = (cache_dir / f.name).with_suffix(".wav")

        if not new.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            if TORCHAUDIO_RESAMPLING:
                x, fsorig = torchaudio.load(str(f))
                x = torch.mean(x,0).unsqueeze(0) # convert to mono
                resampler = torchaudio.transforms.Resample(
                    fsorig,
                    self.ml.sr,
                    lowpass_filter_width=64,
                    rolloff=0.9475937167399596,
                    resampling_method="sinc_interp_kaiser",
                    beta=14.769656459379492,
                )
                y = resampler(x)
                torchaudio.save(new, y, self.ml.sr, encoding="PCM_S", bits_per_sample=16)
            else:                
                sox_args = ['-r', str(self.ml.sr), '-c', '1', '-b', '16']
    
                # ffmpeg has bad resampling compared to SoX
                # SoX has bad format support compared to ffmpeg
                # If the file format is not supported by SoX, use ffmpeg to convert it to wav
    
                if f.suffix[1:] not in self.sox_formats:
                    # Use ffmpeg for format conversion and then pipe to sox for resampling
                    with tempfile.TemporaryDirectory() as tmp:
                        tmp = Path(tmp) / 'temp.wav'
    
                        # Open ffmpeg process for format conversion
                        subprocess.run([
                            ffmpeg_path, 
                            "-hide_banner", "-loglevel", "error", 
                            "-i", f, tmp])
                        
                        # Open sox process for resampling, taking input from ffmpeg's output
                        subprocess.run([sox_path, tmp, *sox_args, new])
                        
                else:
                    # Use sox for resampling
                    subprocess.run([sox_path, f, *sox_args, new])

        return self.ml.load_wav(new)
    
    def read_embedding_file(self, audio_dir: Union[str, Path]):
        """
        Read embedding from a cached file.
        """
        cache = get_cache_embedding_path(self.ml.name, audio_dir)
        assert cache.exists(), f"Embedding file {cache} does not exist, please run cache_embedding_file first."
        return np.load(cache)
    
    def cache_embedding_file(self, audio_dir: Union[str, Path]):
        """
        Compute embedding for an audio file and cache it to a file.
        """
        cache = get_cache_embedding_path(self.ml.name, audio_dir)

        if cache.exists():
            return

        # Load file, get embedding, save embedding
        wav_data = self.load_audio(audio_dir)
        embd = self.ml.get_embedding(wav_data)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, embd)

    def embedding_mse(self, baseline: PathLike, eval_dir: PathLike, csv_name: Union[Path, str]) -> Path:
        """
        Calculate the FAD score for each individual file in eval_dir and write the results to a csv file.

        :param baseline: Baseline matrix or directory containing baseline audio files
        :param eval_dir: Directory containing eval audio files
        :param csv_name: Name of the csv file to write the results to
        :return: Path to the csv file
        """
        csv = Path(csv_name)
        if isinstance(csv_name, str):
            csv = Path('data') / f'fad-individual' / self.ml.name / csv_name
        if csv.exists():
            log.info(f"CSV file {csv} already exists, exiting...")
            return csv

        # 2. Define helper function for calculating z score
        def _find_z_helper(f,f_ref):
            try:
                # Calculate FAD for individual songs
                embd_ref = self.read_embedding_file(f_ref)
                embd = self.read_embedding_file(f)
                mse = np.mean((embd_ref-embd)**2)
                return mse

            except Exception as e:
                traceback.print_exc()
                log.error(f"An error occurred calculating individual FAD using model {self.ml.name} on file {f}")
                log.error(e)

        # 3. Calculate z score for each eval file
        _files = list(Path(eval_dir).glob("*.*"))
        _files.sort()
        _files_ref = list(Path(baseline).glob("*.*"))
        _files_ref.sort()
        # Check if order is correct ==> files_ref should be the same as files
        for file,file_ref in zip(_files, _files_ref):
            file_id_ref = file_ref.stem.split("_")[-1]
            file_id = file.stem.split("_")[-1]
            assert file_id == file_id_ref, f"File {file} and {file_ref} do not match. Please check the order of the files."


        scores = tmap(_find_z_helper, _files, _files_ref, desc=f"Calculating scores", max_workers=self.audio_load_worker)

        # 4. Write the sorted z scores to csv
        pairs = list(zip(_files, scores))
        pairs = [p for p in pairs if p[1] is not None]
        pairs = sorted(pairs, key=lambda x: np.abs(x[1]))
        write(csv, "\n".join([",".join([str(x).replace(',', '_') for x in row]) for row in pairs]))

        return csv