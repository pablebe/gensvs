import os
from argparse import ArgumentParser
from gensvs import EmbeddingMSE, get_all_models, cache_embedding_files
from pathlib import Path

WORKERS = 1

models = {m.name: m for m in get_all_models()}


def main():
    parser = ArgumentParser()

    parser.add_argument('--test-dir', type=str, required=True, help='Path to processed audio directory')
    parser.add_argument('--target-dir', type=str, required=True, help='Path to target audio directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to output directory for metric results')
    parser.add_argument('--embedding', type=str, default='MERT-v1-95M', choices=models.keys())
    parser.add_argument('--workers', type=int, default=WORKERS, help='Number of workers for embedding calculation. Default: 1')

    args = parser.parse_args() 
    
    model = models[args.embedding]
       
    # 1. Calculate embedding files for each dataset
    for d in [args.target_dir, args.test_dir]:
        if Path(d).is_dir():
            cache_embedding_files(d, model, workers=args.workers)
   
    csv_out_path = Path(os.path.join(args.output_dir,args.embedding+'_MSE', 'embd_mse.csv'))


    emb_mse = EmbeddingMSE(model, audio_load_worker=args.workers, load_model=False)
    emb_mse.embedding_mse(args.target_dir, args.test_dir, csv_out_path)


if __name__ == "__main__":
    main()

