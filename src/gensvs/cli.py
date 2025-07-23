import argparse
import sys
from .models import SGMSVS, MelRoFoBigVGAN


def main():
    parser = argparse.ArgumentParser(description="Inference of generative SVS models.")
    parser.add_argument("--model", type=str, required=True, choices=['sgmsvs', 'melrofobigvgan'], help="The model to use for inference.")
    parser.add_argument("--mix-dir", type=str, required=True, help="Directory containing the musical mixtures.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the separated vocal files.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device to run the inference on (default: cuda).")
    parser.add_argument("--loudness-normalize", action="store_true", help="Whether to apply loudness normalization to the output files.")
    parser.add_argument("--loudness-level", type=float, default=-18.0, help="Target loudness level in LUFS for normalization (default: -18.0).")
    parser.add_argument("--output-mono", action="store_true", help="If set output will contain two channels with identical signals.")
    parser.add_argument("--sampler-type", type=str, default="pc", choices=["pc", "ode"], help="Type of sampler to use (default: pc).")
    parser.add_argument("--corrector", type=str, default="ald", choices=["ald", "langevin", "none"], help="Corrector type to use (default: ald).")
    parser.add_argument("--corrector-steps", type=int, default=2, help="Number of corrector steps (default: 2).")
    parser.add_argument("--N", type=int, default=45, help="Number of inference steps (default: 45).")
    parser.add_argument("--snr", type=float, default=0.5, help="Signal-to-noise ratio for inference (default: 0.5).")
    parser.add_argument("--random-seed", type=int, default=1234, help="Random seed for reproducibility (default: 1234).")
    args = parser.parse_args()
    
    
    
    
    if args.model == 'sgmsvs':
        model = SGMSVS()
        model.run_folder(
            test_dir=args.mix_dir,
            out_dir=args.output_dir,
            sampler_type=args.sampler_type,
            corrector=args.corrector,
            corrector_steps=args.corrector_steps,
            N=args.N,
            snr=args.snr,
            random_seed=args.random_seed,
            loudness_normalize=args.loudness_normalize,
            loudness_level=args.loudness_level,
            output_mono=args.output_mono
        )
    elif args.model == 'melrofobigvgan':
        model = MelRoFoBigVGAN()
        model.run_folder(test_dir=args.mix_dir,
                         out_dir=args.output_dir,
                         loudness_normalize=args.loudness_normalize,
                         loudness_level=args.loudness_level,
                         output_mono=args.output_mono)
        
        
if __name__ == "__main__":
    sys.exit(main())