# gensvs: Generative Singing Voice Separation
This Python package supports the paper "Towards Reliable Objective Evaluation Metrics for Generative Singing Voice Separation" by Paul A. Bereuter, Benjamin Stahl, Mark D. Plumbley and Alois Sontacchi, presented at WASPAA 2025.
It facilitates the straightforward inference of the two proposed generative models (SGMSVS and MelRoFo (S) + BigVGAN) and the computation of the embedding MSE metrics that exhibited the highest correlation with human DMOS ratings. 

Additionally, this package comprises all packages to execute the [training code available at GitHub](https://github.com/pablebe/gensvs_eval).

> Note: When using this package to carry out inference or evaluation, the necessary models (e.g. singing voice separation or embedding models) are downloaded automatically.

## 🚀 Installation and Usage
### Installation via pip
You can install the package via pip using:
```bash
pip install gensvs
```
### Installation from Source
The package was tested on Debian but should also work with CUDA support on Microsoft Windows, if you follow the steps below.
1. Clone this repository 
2. Run ```pip install "."```
### Installation on Microsoft Windows 
1. Install the package via pip or from Source (see above)
2. Reinstall PyTorch with CUDA>=12.6 using install command from ["PyTorch - Get Started"](https://pytorch.org/get-started/locally/) to get CUDA support.
### Setting up a conda environment using the provided bash script
We recommend installing this package in a separate conda environment. The recommended settings for the conda environment can be found in the accompanying [.yml file](https://github.com/pablebe/gensvs/blob/master/env_info/gensvs_env.yml). If you have a running conda installation (e.g. [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) or [Miniforge](https://github.com/conda-forge/miniforge)) and are working on a Linux system, you can run the included [Bash Script](https://github.com/pablebe/gensvs/blob/master/env_info/setup_gensvs_env.sh) from the root directory to create the conda environment and install the package. This bash script will automatically create a conda environment, install the ```gensvs``` package via pip, and delete the subfolders in the cache folder ```~/.cache/torch_extensions```. These subfolders can sometimes prevent the inference of the SGMSVS model.
Further information on the usage of model inference and model evaluation is provided below.
## 🏃🏽‍♀️‍➡️ Model Inference
### Command Line Tool
You can carry out the model inference using our command line tool. An example call for the SGMSVS model is shown below:
```bash
gensvs --model sgmsvs --device cuda  --mix-dir audio_examples/mixture --output-dir audio_examples/separated --output-mono
``` 
To isolate the inference to one CUDA device please you can set the environment variable ```CUDA_VISIBLE_DEVICES``` before the calling the command tool. An exemplary inference call for 'MelRoFo (S) + BigVGAN', isolated on GPU 0, can be found below:
```bash
CUDA_VISIBLE_DEVICES=0 gensvs --model melrofobigvgan --device cuda  --mix-dir audio_examples/mixture --output-dir audio_examples/separated --output-mono
``` 
> Note: when using 'MelRoFo (S) + BigVGAN' model to separate vocals, the signals before (only MelRoFo (S) separation) and after the finetuned BigVGAN are saved into the output directory.

For more details on the available inference parameters please call:
```bash 
gensvs --help
```  
### Model Inference from Python script
You can also import the models from the package to carry out the inference on a folder of musical mixtures from a Python script.
```Python
from gensvs import MelRoFoBigVGAN, SGMSVS

MIX_PATH = './audio_examples/mixture'
SEP_PATH = './audio_examples/separated'

sgmsvs_model = SGMSVS()
melrofo_model = MelRoFoBigVGAN()

sgmsvs_model.run_folder(MIX_PATH, SEP_PATH, loudness_normalize=False, loudness_level=-18, output_mono=True, ch_by_ch_processing=False)
melrofo_model.run_folder(MIX_PATH, SEP_PATH, loudness_normalize=False, loudness_level=-18, output_mono=True)
```

You can find this script on the GitHub-Repository in [```./demo/inference_demo.py```](https://github.com/pablebe/gensvs/tree/master/demo).

## 📈 Model Evaluation with Embedding-based MSE
In this package, we have included the calculation of the proposed embedding MSEs from the paper, building on the code published with Microsoft's [Frechet Audio Distance Tookit](https://github.com/microsoft/fadtk/tree/main). The Mean Squared Error on either [MERT](https://huggingface.co/m-a-p/MERT-v1-95M) or [Music2Latent](https://github.com/SonyCSLParis/music2latent) embeddings can be calculated with the command line tool or a Python script.
### Command Line Tool
An example command line call to calculate the MSE on [MERT](https://huggingface.co/m-a-p/MERT-v1-95M) embeddings is shown below:
```bash
gensvs-eval --test-dir ./demo/audio_examples/separated/sgmsvs --target-dir ./demo/audio_examples/target --output-dir ./demo/results/sgmsvs --embedding MERT-v1-95M 
```
For more details on the available flags please call:
```bash 
gensvs-eval --help
```  
### Model Evaluation from Python script 
To calculate the embedding MSE from a Python script you can use:
```Python
import os
from gensvs import EmbeddingMSE, get_all_models, cache_embedding_files
from pathlib import Path

#embedding calculation builds on multiprocessing library => don't forget to wrap your code in a main function
WORKERS = 8

SEP_PATH = './demo/audio_examples/separated'
TGT_PATH = './demo/audio_examples/target'
OUT_DIR = './demo/eval_metrics_demo'

def main():
    # calculate embedding MSE
    embedding = 'MERT-v1-95M'#music2latent
    models = {m.name: m for m in get_all_models()}
    model = models[embedding]
    svs_model_names = ['sgmsvs', 'melroformer_bigvgan', 'melroformer_small']

    for model_name in svs_model_names:
        # 1. Calculate and store embedding files for each dataset
        for d in [TGT_PATH, os.path.join(SEP_PATH, model_name)]:
            if Path(d).is_dir():
                cache_embedding_files(d, model, workers=WORKERS, load_model=True)

        csv_out_path = Path(os.path.join(OUT_DIR, model_name,embedding+'_MSE', 'embd_mse.csv'))
        # 2. Calculate embedding MSE for each file in folder
        emb_mse = EmbeddingMSE(model, audio_load_worker=WORKERS, load_model=False)
        emb_mse.embedding_mse(TGT_PATH, os.path.join(SEP_PATH, model_name), csv_out_path)


if __name__ == "__main__":
    main()
```
This script can be found in [```./demo/evaluation_demo.py```](https://github.com/pablebe/gensvs/tree/master/demo)
## ℹ️ Further information
- Paper: [Preprint](https://arxiv.org/pdf/2507.11427)
- Website: [Companion Page](https://pablebe.github.io/gensvs_eval_companion_page/) 
- Data: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15911723.svg)](https://doi.org/10.5281/zenodo.15911723)
- Model Checkpoints: [Hugging Face](https://huggingface.co/collections/pablebe/gensvs-eval-model-checkpoints-687e1c967b43f867f34d6225)
- More Code: [GitHub](https://github.com/pablebe/gensvs_eval)

## Citations, References and Acknowledgements
If you use this package in your work please do not forget to cite our paper and the work which built the foundation for this package.
Our paper can be cited with:
```bib
@misc{bereuter2025,
      title={Towards Reliable Objective Evaluation Metrics for Generative Singing Voice Separation Models}, 
      author={Paul A. Bereuter and Benjamin Stahl and Mark D. Plumbley and Alois Sontacchi},
      year={2025},
      eprint={2507.11427},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2507.11427}, 
}
```
The inference code for the SGMSVS model was built upon the code made available in:
```bib
@article{richter2023speech,
         title={Speech Enhancement and Dereverberation with Diffusion-based Generative Models},
         author={Richter, Julius and Welker, Simon and Lemercier, Jean-Marie and Lay, Bunlong and Gerkmann, Timo},
         journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
         volume={31},
         pages={2351-2364},
         year={2023},
         doi={10.1109/TASLP.2023.3285241}
        }
```
The inference code for MelRoFo (S) + BigVGAN was put together from the code available at:
```bib
@misc{solovyev2023benchmarks,
      title={Benchmarks and leaderboards for sound demixing tasks}, 
      author={Roman Solovyev and Alexander Stempkovskiy and Tatiana Habruseva},
      year={2023},
      eprint={2305.07489},
      archivePrefix={arXiv},
      howpublished={\url{https://github.com/ZFTurbo/Music-Source-Separation-Training}},
      primaryClass={cs.SD},
      url={https://github.com/ZFTurbo/Music-Source-Separation-Training}
      }
```
```bib
@misc{jensen2024melbandroformer,
      author       = {Kimberley Jensen},
      title        = {Mel-Band-Roformer-Vocal-Model},
      year         = {2024},
      howpublished = {\url{https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model}},
      note         = {GitHub repository},
      url          = {https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model}
    }
```
```bib
@inproceedings{lee2023bigvgan,
               title={BigVGAN: A Universal Neural Vocoder with Large-Scale Training},
               author={Sang-gil Lee and Wei Ping and Boris Ginsburg and Bryan Catanzaro and Sungroh Yoon},
               booktitle={in Proc. ICLR, 2023},
               year={2023},
               url={https://openreview.net/forum?id=iTtGCMDEzS_}
              }
```
The whole evaluation code was created using Microsoft's [Frechet Audio Distance Tookit](https://github.com/microsoft/fadtk/tree/main) as a template
```bib
@inproceedings{fadtk,
               title = {Adapting Frechet Audio Distance for Generative Music Evaluation},
               author = {Azalea Gui, Hannes Gamper, Sebastian Braun, Dimitra Emmanouilidou},
               booktitle = {Proc. IEEE ICASSP 2024},
               year = {2024},
               url = {https://arxiv.org/abs/2311.01616},
              }
```
If you use the [MERT](https://huggingface.co/m-a-p/MERT-v1-95M) or [Music2Latent](https://github.com/SonyCSLParis/music2latent) MSE please also cite the initial work in which the embeddings were proposed. 

For [MERT](https://huggingface.co/m-a-p/MERT-v1-95M):
```bib
@misc{li2023mert,
      title={MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training}, 
      author={Yizhi Li and Ruibin Yuan and Ge Zhang and Yinghao Ma and Xingran Chen and Hanzhi Yin and Chenghua Lin and Anton Ragni and Emmanouil Benetos and Norbert Gyenge and Roger Dannenberg and Ruibo Liu and Wenhu Chen and Gus Xia and Yemin Shi and Wenhao Huang and Yike Guo and Jie Fu},
      year={2023},
      eprint={2306.00107},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
For [Music2Latent](https://github.com/SonyCSLParis/music2latent):
```bib
@inproceedings{pasini2024music2latent,
  author       = {Marco Pasini and Stefan Lattner and George Fazekas},
  title        = {{Music2Latent}: Consistency Autoencoders for Latent Audio Compression},
  booktitle    = ismir,
  year         = 2024,
  pages        = {111-119},
  venue        = {San Francisco, California, USA and Online},
  doi          = {10.5281/zenodo.14877289},
}
```
# License
In accordance with Microsoft's [Frechet Audio Distance Tookit](https://github.com/microsoft/fadtk/tree/main) this work is made available under an 
[MIT License](https://github.com/pablebe/gensvs/blob/master/LICENSE).

