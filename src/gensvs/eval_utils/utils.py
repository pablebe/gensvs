
import subprocess
from typing import Union
from pathlib import Path

PathLike = Union[str, Path]

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

