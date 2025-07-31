import os
from gensvs import EmbeddingMSE, get_all_models, cache_embedding_files
from pathlib import Path

#embedding calculation builds on multiprocessing library => don't forget to wrap your code in a main function
WORKERS = 8

SEP_PATH = './demo/audio_examples/separated'
TGT_PATH = './demo/audio_examples/target'
OUT_DIR = './demo/eval_metrics'

def main():
    # calculate embedding MSE
    embedding = 'music2latent'#'MERT-v1-95M'#music2latent
    models = {m.name: m for m in get_all_models()}
    model = models[embedding]
    svs_model_names = ['sgmsvs', 'melroformer_bigvgan', 'melroformer_small']

    for model_name in svs_model_names:
        # 1. Calculate and store embedding files for each dataset
        for d in [TGT_PATH, os.path.join(SEP_PATH, model_name)]:
            if Path(d).is_dir():
                cache_embedding_files(d, model, workers=WORKERS)

        csv_out_path = Path(os.path.join(OUT_DIR, model_name,embedding+'_MSE', 'embd_mse.csv'))
        # 2. Calculate embedding MSE for each file in folder
        emb_mse = EmbeddingMSE(model, audio_load_worker=WORKERS, load_model=False)
        emb_mse.embedding_mse(TGT_PATH, os.path.join(SEP_PATH, model_name), csv_out_path)


if __name__ == "__main__":
    main()