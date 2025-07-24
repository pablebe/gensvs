
import numpy as np
import traceback
import torch
import torchaudio
import os
import multiprocessing
import soundfile
import subprocess
import shutil
import tempfile
import torch.serialization
import _codecs

from abc import ABC, abstractmethod
from typing import Union
from pathlib import Path
from hypy_utils import write
from hypy_utils.logging_utils import setup_logger
from hypy_utils.tqdm_utils import tmap

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

class ModelLoader(ABC):
    """
    Abstract class for loading a model and getting embeddings from it. The model should be loaded in the `load_model` method.
    """
    def __init__(self, name: str, num_features: int, sr: int):
        self.model = None
        self.sr = sr
        self.num_features = num_features
        self.name = name
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def get_embedding(self, audio: np.ndarray):
        embd = self._get_embedding(audio)
        if self.device == torch.device('cuda'):
            embd = embd.cpu()
        embd = embd.detach().numpy()
        
        # If embedding is float32, convert to float16 to be space-efficient
        if embd.dtype == np.float32:
            embd = embd.astype(np.float16)

        return embd

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def _get_embedding(self, audio: np.ndarray):
        """
        Returns the embedding of the audio file. The resulting vector should be of shape (n_frames, n_features).
        """
        pass

    def load_wav(self, wav_file: Path):
        wav_data, _ = soundfile.read(wav_file, dtype='int16')
        wav_data = wav_data / 32768.0  # Convert to [-1.0, +1.0]

        return wav_data


def get_all_models() -> list[ModelLoader]:
    ms = [
         Music2LatentModel(),
        *(MERTModel(layer=v) for v in range(1, 13)),
    ]

    return ms

def find_sox_formats(sox_path: str) -> list[str]:
    """
    Find a list of file formats supported by SoX
    """
    try:
        out = subprocess.check_output((sox_path, "-h")).decode()
        return substr_between(out, "AUDIO FILE FORMATS: ", "\n").split()
    except:
        return []

def substr_between(s: str, start: str | None = None, end: str | None = None):
    """
    Get substring between two strings

    >>> substr_between('abc { meow } def', '{', '}')
    ' meow '
    """
    if start:
        s = s[s.index(start) + len(start):]
    if end:
        s = s[:s.index(end)]
    return s

def get_cache_embedding_path(model: str, audio_dir: PathLike) -> Path:
    """
    Get the path to the cached embedding npy file for an audio file.

    :param model: The name of the model
    :param audio_dir: The path to the audio file
    """
    audio_dir = Path(audio_dir)
    return audio_dir.parent / "embeddings" / model / audio_dir.with_suffix(".npy").name

def _cache_embedding_batch(args):
    fs: list[Path]
    ml: ModelLoader
    fs, ml, kwargs = args
    embmse = EmbeddingMSE(ml, **kwargs)
    for f in fs:
        log.info(f"Loading {f} using {ml.name}")
        embmse.cache_embedding_file(f)


def cache_embedding_files(files: Union[list[Path], str, Path], ml: ModelLoader, workers: int = 8, **kwargs):
    """
    Get embeddings for all audio files in a directory.

    :param ml_fn: A function that returns a ModelLoader instance.
    """
    if isinstance(files, (str, Path)):
        files = list(Path(files).glob('*.*'))

    # Filter out files that already have embeddings
    files = [f for f in files if not get_cache_embedding_path(ml.name, f).exists()]
    if len(files) == 0:
        log.info("All files already have embeddings, skipping.")
        return

    log.info(f"[Embedding MSE] Loading {len(files)} audio files...")

    # Split files into batches
    batches = list(np.array_split(files, workers))
    
    # Cache embeddings in parallel
    multiprocessing.set_start_method('spawn', force=True)
    with torch.multiprocessing.Pool(workers) as pool:
        pool.map(_cache_embedding_batch, [(b, ml, kwargs) for b in batches])

class Music2LatentModel(ModelLoader):
    """
    Add a short description of your model here.
    """
    def __init__(self):
        # Define your sample rate and number of features here. Audio will automatically be resampled to this sample rate.
        super().__init__("music2latent", num_features=64, sr=44100)
        # Add any other variables you need here

    def load_model(self):
        import music2latent
        import numpy
        import torch.serialization
        from music2latent import EncoderDecoder
        torch.serialization.add_safe_globals([numpy.dtype,numpy.dtypes.Float64DType,numpy.core.multiarray.scalar])
        self.model = EncoderDecoder()

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        # Calculate the embeddings using your model
        latent = self.model.encode(audio)
        latent = latent.squeeze().transpose(0, 1)  # [timeframes, 64]
        return latent

    def load_wav(self, wav_file: Path):
        # Optionally, you can override this method to load wav file in a different way. The input wav_file is already in the correct sample rate specified in the constructor.
        return super().load_wav(wav_file)
    
    
class MERTModel(ModelLoader):
    """
    MERT model from https://huggingface.co/m-a-p/MERT-v1-330M

    Please specify the layer to use (1-12).
    """
    def __init__(self, size='v1-95M', layer=12, limit_minutes=6):
        super().__init__(f"MERT-{size}" + ("" if layer == 12 else f"-{layer}"), 768, 24000)
        self.huggingface_id = f"m-a-p/MERT-{size}"
        self.layer = layer
        self.limit = limit_minutes * 60 * self.sr
        
    def load_model(self):
        from transformers import Wav2Vec2FeatureExtractor
        from transformers import AutoModel
        
        self.model = AutoModel.from_pretrained(self.huggingface_id, trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.huggingface_id, trust_remote_code=True)
        # self.sr = self.processor.sampling_rate
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        # Limit to 9 minutes
        if audio.shape[0] > self.limit:
            log.warning(f"Audio is too long ({audio.shape[0] / self.sr / 60:.2f} minutes > {self.limit / self.sr / 60:.2f} minutes). Truncating.")
            audio = audio[:self.limit]

        inputs = self.processor(audio, sampling_rate=self.sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            out = torch.stack(out.hidden_states).squeeze() # [13 layers, timeframes, 768]
            out = out[self.layer] # [timeframes, 768]

        return out

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