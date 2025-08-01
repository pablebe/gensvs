import torch
import soundfile
import logging
import numpy as np
import numpy
import torch.serialization
torch.serialization.add_safe_globals([numpy.dtype,numpy.dtypes.Float64DType,numpy.core.multiarray.scalar])
from abc import ABC, abstractmethod
from pathlib import Path

log = logging.getLogger(__name__)


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


class Music2LatentModel(ModelLoader):
    """
    Add a short description of your model here.
    """
    def __init__(self):
        # Define your sample rate and number of features here. Audio will automatically be resampled to this sample rate.
        super().__init__("music2latent", num_features=64, sr=44100)
        from music2latent.utils import download_model
        #ensure model is downloaded before parallelization starts
        download_model()
        # Add any other variables you need here

    def load_model(self):
        from music2latent import EncoderDecoder
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

def get_all_models() -> list[ModelLoader]:
    ms = [
         Music2LatentModel(),
        *(MERTModel(layer=v) for v in range(1, 13)),
    ]

    return ms