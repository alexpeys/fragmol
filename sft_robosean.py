import os
import argparse
import torch
import torch.distributed as dist
import random
import numpy as np
import time
import glob
import subprocess
import tempfile
import pandas as pd
from utils.models import EmbeddingConditionalLlamaDecoder, create_custom_llama_config
from synthesis.dataloader import SynthesisPathDataset, property_tokens
from synthesis.helpers import get_reactions, execute_path

from s3torchconnector import S3Checkpoint

from datetime import timedelta
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb


def generate_and_eval_with_classifier(
    model,
    tokenizer,
    reactions,
    classifier_path,
    train_smiles_set,
    train_fingerprints,
    num_molecules=1024,
    temperature=1.0,
    top_p=0.95,
    max_tokens=200,
    device='cuda',
    step=0,
    classifier_batch_size=512,
):
    """
    Generate molecules unconditionally, run external classifier, return mean score.

    Args:
        model: The EmbeddingConditionalLlamaDecoder model
        tokenizer: The synthesis tokenizer
        reactions: Reactions dict from get_reactions()
        classifier_path: S3 path to classifier folder
        train_smiles_set: Set of canonical SMILES from training data (for novelty check)
        train_fingerprints: List of Morgan fingerprints from training data
        num_molecules: Number of molecules to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens to generate
        device: Device to run generation on
        step: Current training step (for logging)
        classifier_batch_size: Batch size for classifier inference

    Returns:
        dict with generation stats and classifier scores
    """
    from tqdm import tqdm
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem import AllChem
    from rdkit import DataStructs

    model.eval()

    # Get special token IDs
    bos_token_id = tokenizer.convert_tokens_to_ids('[BOS]')
    eos_token_id = tokenizer.convert_tokens_to_ids('[EOS]')

    # Generate in batches to avoid OOM
    gen_batch_size = 128
    generated_sequences = []

    print(f"\nGenerating {num_molecules} synthesis paths (unconditional) in batches of {gen_batch_size}...")

    for batch_start in range(0, num_molecules, gen_batch_size):
        batch_end = min(batch_start + gen_batch_size, num_molecules)
        batch_size = batch_end - batch_start

        input_ids_list = []
        for _ in range(batch_size):
            tokens = [bos_token_id]
            input_ids_list.append(torch.tensor(tokens, device=device))

        with torch.no_grad():
            batch_sequences = model.generate(
                input_ids_list=input_ids_list,
                conditioning_embeddings=None,
                max_generated_tokens=max_tokens,
                stop_token_id=eos_token_id,
                temperature=temperature,
                top_p=top_p,
            )
        generated_sequences.extend(batch_sequences)
        print(f"  Generated {len(generated_sequences)}/{num_molecules}...")

    print(f"Generation complete, got {len(generated_sequences)} sequences")

    # Decode, parse, and execute synthesis paths
    valid_molecules = []
    valid_smiles = []
    invalid_count = 0
    parse_errors = 0
    execution_errors = 0

    print(f"Decoding, parsing, and executing synthesis paths...")
    for idx, seq in enumerate(tqdm(generated_sequences, desc="Validating", unit="path")):
        try:
            seq_list = seq.tolist() if hasattr(seq, 'tolist') else seq
            decoded_full = tokenizer.decode(seq_list)

            # Remove special tokens and spaces
            clean_str = decoded_full.replace(' ', '')
            for tok in ['[BOS]', '[EOS]', '[PAD]'] + property_tokens:
                clean_str = clean_str.replace(tok, '')
            clean_str = clean_str.strip()

            # Check if valid synthesis path
            if not clean_str.startswith('<ADD>'):
                parse_errors += 1
                invalid_count += 1
                continue

            # Execute synthesis path
            result, _ = execute_path(clean_str, reactions)

            if result is None:
                execution_errors += 1
                invalid_count += 1
                continue

            # Validate with RDKit
            mol = Chem.MolFromSmiles(result)
            if mol is None:
                invalid_count += 1
                continue

            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            valid_molecules.append(mol)
            valid_smiles.append(canonical_smiles)

        except Exception as e:
            invalid_count += 1
            continue

    num_valid = len(valid_molecules)
    validity_rate = num_valid / num_molecules if num_molecules > 0 else 0

    print(f"Valid molecules: {num_valid}/{num_molecules} ({validity_rate*100:.1f}%)")
    print(f"  Parse errors: {parse_errors}")
    print(f"  Execution errors: {execution_errors}")

    # Run classifier on valid molecules
    mean_score = 0.0
    std_score = 0.0
    p90_score = 0.0
    p99_score = 0.0
    scores = []

    if num_valid > 0 and classifier_path:
        print(f"Running classifier on {num_valid} molecules...")

        # Write molecules to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            tmp_input_path = f.name
            df = pd.DataFrame({'smiles': valid_smiles})
            df.to_csv(f, index=False)

        tmp_output_path = tmp_input_path.replace('.csv', '_with_scores.csv')

        try:
            # Run classifier
            cmd = [
                'python', '/home/ubuntu/lolmol_inference_server/run_classifier.py',
                '--classifier_path', classifier_path,
                '--eval_data_path', tmp_input_path,
                '--batch_size', str(classifier_batch_size),
            ]
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd='/home/ubuntu/lolmol_inference_server')

            if result.returncode != 0:
                print(f"Classifier error: {result.stderr}")
            else:
                # Read scores
                scores_df = pd.read_csv(tmp_output_path)
                scores = scores_df['logit_mean'].tolist()
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                p90_score = np.percentile(scores, 90)
                p99_score = np.percentile(scores, 99)
                print(f"Classifier scores: mean={mean_score:.4f}, std={std_score:.4f}, p90={p90_score:.4f}, p99={p99_score:.4f}")

        except Exception as e:
            print(f"Error running classifier: {e}")

        finally:
            # Cleanup temp files
            if os.path.exists(tmp_input_path):
                os.remove(tmp_input_path)
            if os.path.exists(tmp_output_path):
                os.remove(tmp_output_path)

    # Check novelty: fraction of generated mols with Tanimoto > 0.95 to training set
    frac_memorized = 0.0
    num_memorized = 0
    if num_valid > 0 and train_fingerprints:
        print(f"Checking novelty against {len(train_fingerprints)} training molecules...")
        for mol in tqdm(valid_molecules, desc="Novelty check", unit="mol"):
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                # Get max similarity to any training molecule
                sims = DataStructs.BulkTanimotoSimilarity(fp, train_fingerprints)
                max_sim = max(sims) if sims else 0.0
                if max_sim > 0.95:
                    num_memorized += 1
            except:
                continue
        frac_memorized = num_memorized / num_valid
        print(f"Memorization check: {num_memorized}/{num_valid} ({frac_memorized*100:.1f}%) have Tanimoto > 0.95 to training set")

    # Log to wandb
    log_dict = {
        "validity_rate": validity_rate,
        "parse_error_rate": parse_errors / num_molecules if num_molecules > 0 else 0,
        "execution_error_rate": execution_errors / num_molecules if num_molecules > 0 else 0,
        "num_valid": num_valid,
        "classifier_mean_score": mean_score,
        "classifier_std_score": std_score,
        "classifier_p90_score": p90_score,
        "classifier_p99_score": p99_score,
        "frac_memorized": frac_memorized,
    }

    # Log some molecule images
    if num_valid > 0:
        mols_to_visualize = valid_molecules[:25]
        smiles_to_log = valid_smiles[:25]
        images = []
        for mol, smi in zip(mols_to_visualize, smiles_to_log):
            try:
                img = Draw.MolToImage(mol, size=(300, 300))
                img_array = np.array(img)
                images.append(wandb.Image(img_array, caption=smi[:50]))
            except:
                continue
        if images:
            log_dict["generated_molecules"] = images

    wandb.log(log_dict, step=step)

    model.train()

    return {
        "num_valid": num_valid,
        "validity_rate": validity_rate,
        "parse_errors": parse_errors,
        "execution_errors": execution_errors,
        "mean_score": mean_score,
        "std_score": std_score,
        "p90_score": p90_score,
        "p99_score": p99_score,
        "frac_memorized": frac_memorized,
        "scores": scores,
        "valid_smiles": valid_smiles,
    }


def setup_ddp():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Not using distributed training")
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend='nccl', timeout=timedelta(hours=1))
        torch.cuda.set_device(local_rank)

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    return rank, world_size, local_rank, device

def cleanup_ddp():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description="SFT for molecule generation with classifier eval")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataset", type=str, default='synthesis/eval_set/seno_zinc_examples.parquet')
    parser.add_argument("--classifier_path", type=str, default='s3://shvaibackups/lolmol_classifiers/seno_screen/',
                        help="S3 path to classifier folder for eval")
    parser.add_argument("--classifier_batch_size", type=int, default=512, help="Batch size for classifier inference")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1, help="Number of dataloader workers")
    parser.add_argument("--checkpoint", type=str, default='s3://shvaibackups/robosean/500m_robosean_molprefix2/75001.pt')
    parser.add_argument("--compile", action="store_true", help="Compile model with torch.compile")

    parser.add_argument("--max_lr", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=7)
    parser.add_argument("--max_grad_norm", type=float, default=1.5)

    parser.add_argument("--max_len", type=int, default=200, help="Input len")

    parser.add_argument("--num_generate", type=int, default=1024, help="Number of molecules to generate for eval")
    parser.add_argument("--generation_temperature", type=float, default=1.0, help="Temperature for molecule generation")
    parser.add_argument("--generation_top_p", type=float, default=0.9, help="Top-p for molecule generation")

    ## 100m model: (emb 512, layers 24, heads 8, intermediate_size_multiplier 4)
    ## 200m model: (emb 768, layers 28, heads 12, intermediate_size_multiplier 3)
    ## 300m model: (emb 768, layers 32, heads 12, intermediate_size_multiplier 4)
    ## 500m model: (emb 1024, layers 36, heads 16, intermediate_size_multiplier 3)
    ## esmc 600m model: emb 1152, heads 18, layers 36, intermediate_size_multiplier 8/3
    ## 1b model: (emb 1280, layers 40, heads 20, intermediate_size_multiplier 4)
    ## 1.8b model: (emb 1536, layers 48, heads 24, intermediate_size_multiplier 4)
    parser.add_argument("--emb_dim", type=int, default=1024)
    parser.add_argument("--intermediate_size_multiplier", type=float, default=4)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--num_attention_heads", type=int, default=16)

    parser.add_argument("--local_rank", type=int, default=-1, metavar="N", help="Local process rank.")

    parser.add_argument("--experiment", type=str, default='')

    return parser.parse_args()


def main():
    # Setup DDP
    rank, world_size, local_rank, device = setup_ddp()

    seed = time.time_ns() + rank

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32 - 1))
    
    # Parse arguments
    args = parse_args()

    # Load reactions for synthesis path generation/execution
    reactions = get_reactions()

    # Load tokenizer to get vocab size and special token IDs
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained("synthesis/synthesis_tokenizer")

    # Create model config for molecules
    # Use max vocab id + 1 to handle gaps in vocab (e.g., ID 4 is unused)
    vocab_size = max(tokenizer.get_vocab().values()) + 1
    decoder_config = create_custom_llama_config(
        vocab_size=vocab_size,
        hidden_size=args.emb_dim,
        intermediate_size=int(args.emb_dim * args.intermediate_size_multiplier),
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        pad_token_id=tokenizer.convert_tokens_to_ids('[PAD]'),
        bos_token_id=tokenizer.convert_tokens_to_ids('[BOS]'),
        eos_token_id=tokenizer.convert_tokens_to_ids('[EOS]'),
        cls_token_id=tokenizer.convert_tokens_to_ids('[BOS]'),  # Use BOS as CLS
        max_position_embeddings=args.max_len,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        attention_dropout=0.0,
        rope_base=10_000,
        use_rope=True,
    )

    model = EmbeddingConditionalLlamaDecoder(decoder_config)

    # Print model info on rank 0
    if rank == 0:
        from utils.models import count_parameters
        num_params = count_parameters(model)
        print(f"\nModel Configuration:")
        print(f"  Vocab size: {decoder_config.vocab_size}")
        print(f"  Hidden size: {decoder_config.hidden_size}")
        print(f"  Intermediate size: {decoder_config.intermediate_size}")
        print(f"  Num layers: {decoder_config.num_hidden_layers}")
        print(f"  Num attention heads: {decoder_config.num_attention_heads}")
        print(f"  Max position embeddings: {decoder_config.max_position_embeddings}")
        print(f"  Total parameters: {num_params:,} ({num_params/1e6:.1f}M)")
        print()

    # Move model to device
    model = model.to(device)

    if args.checkpoint != '':
        print(f"Loading from checkpoint: {args.checkpoint}")
        #checkpoint = torch.load(args.checkpoint)

        if args.checkpoint.startswith('s3://'):
            with S3Checkpoint("us-west-2").reader(args.checkpoint) as reader:
               checkpoint = torch.load(reader, map_location='cpu')
        else:
           checkpoint = torch.load(args.checkpoint, map_location='cpu')

        # Handle compiled model state dict (remove _orig_mod. prefix if present)
        if any(k.startswith('_orig_mod.') for k in checkpoint.keys()):
            print("Detected compiled model checkpoint, removing _orig_mod. prefix...")
            new_checkpoint = {}
            prefix = '_orig_mod.'
            for k, v in checkpoint.items():
                if k.startswith(prefix):
                    new_checkpoint[k.replace(prefix, '', 1)] = v  # Remove prefix only once from start
                else:
                    new_checkpoint[k] = v
            # Delete old checkpoint to free memory
            del checkpoint
            checkpoint = new_checkpoint

        model.load_state_dict(checkpoint)

        # Delete checkpoint to free CPU memory
        del checkpoint
        torch.cuda.empty_cache()

        if rank == 0:
            print("Checkpoint loaded and memory cleared")

    if args.compile:
        model = torch.compile(model)
        
    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Load dataset into memory and calculate epoch sizes
    train_df = pd.read_parquet(args.dataset)
    dataset_size = len(train_df)

    # Calculate steps per epoch and total steps
    samples_per_step = args.batch_size * args.grad_accum_steps * world_size
    steps_per_epoch = dataset_size // samples_per_step
    max_steps = steps_per_epoch * args.num_epochs
    eval_every_steps = steps_per_epoch // 2  # Eval every 0.5 epoch

    if rank == 0:
        print(f"Dataset size: {dataset_size}")
        print(f"Samples per step: {samples_per_step}")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total steps ({args.num_epochs} epochs): {max_steps}")
        print(f"Eval every {eval_every_steps} steps (0.5 epoch)")

    # Create dataset and dataloader
    train_dataset = SynthesisPathDataset(
        file_list=[args.dataset],
        tokenizer=tokenizer,
        max_len=args.max_len,
        add_characterization_tokens=False,
        prefix_mol_probability=0.0,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        #prefetch_factor=2 if args.num_workers > 0 else None,
    )

    train_iterator = iter(train_dataloader)

    # Load training SMILES and precompute fingerprints for novelty check (rank 0 only)
    train_smiles_set = set()
    train_fingerprints = []
    if rank == 0:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        print(f"Loading training SMILES from {args.dataset} for novelty check...")
        try:
            train_df = pd.read_parquet(args.dataset)
            if 'product' in train_df.columns:
                smiles_col = 'product'
            elif 'smiles' in train_df.columns:
                smiles_col = 'smiles'
            else:
                smiles_col = train_df.columns[0]

            for smi in train_df[smiles_col]:
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        canonical = Chem.MolToSmiles(mol, canonical=True)
                        train_smiles_set.add(canonical)
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        train_fingerprints.append(fp)
                except:
                    continue
            print(f"Loaded {len(train_smiles_set)} unique training molecules for novelty check")
        except Exception as e:
            print(f"Warning: Could not load training SMILES for novelty check: {e}")

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=args.max_lr)

    # Create LR scheduler with warmup and linear cooldown
    warmup_steps = 20
    cooldown_steps = max_steps - warmup_steps

    # Warmup scheduler: goes from very small LR to max_lr over warmup_steps
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)

    # Cooldown scheduler: goes from max_lr to min_lr over remaining steps
    cooldown_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=args.min_lr/args.max_lr, total_iters=cooldown_steps)

    # Combine schedulers
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cooldown_scheduler], milestones=[warmup_steps])

    # Initialize wandb only on rank 0
    if rank == 0:
        wandb.init(
            project="robosean-sft",
            name=args.experiment,
            config=vars(args),
        )

    # Training loop
    model.train()
    global_step = 0
    total_start_time = time.time()

    while global_step < max_steps:
        time_start = time.time()
        optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=device)

        for grad_accum_step in range(args.grad_accum_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                batch = next(train_iterator)

            token_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            with torch.autocast('cuda', dtype=torch.bfloat16):
                # For causal language modeling, labels are the same as input_ids
                # The model will shift them internally for next-token prediction
                outputs = model(
                    input_ids=token_ids,
                    attention_mask=None,
                    labels=token_ids,
                )

                step_loss = outputs['loss'] / args.grad_accum_steps

            total_loss = total_loss + step_loss
            step_loss.backward()

        # Synchronize gradients across processes
        if world_size > 1:
            dist.barrier()

        # Clip gradients and step
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        time_taken = time.time() - time_start

        # Logging (only on rank 0)
        if rank == 0:
            total_time_taken = time.time() - total_start_time
            avg_time_per_step = total_time_taken / (global_step + 1)
            current_lr = optimizer.param_groups[0]['lr']

            data_done = (global_step * args.batch_size * args.grad_accum_steps * world_size) / 1e6

            # Build log string for molecule diffusion training
            log_str = f"Step {global_step} | Step Time: {time_taken:.2f} | Avg Time/Step: {avg_time_per_step:.2f} | Loss: {total_loss.item():.4f}  | Grad Norm: {grad_norm.item():.2f} | Current LR: {current_lr:.2e} | Data Seen (M): {data_done:.2f}"
            print(log_str)

            # Build wandb log args for molecule diffusion training
            log_args = {
                "train_loss": total_loss.item(),
                "lr": current_lr,
                "step_time": time_taken,
                "grad_norm": grad_norm.item(),
                "data_seen_millions": data_done,
            }

            wandb.log(log_args, step=global_step)

        # Generate and eval with classifier (every 0.5 epoch)
        if (eval_every_steps > 0 and global_step % eval_every_steps == 0 and global_step > 0) or global_step == 10:
            # Synchronize all ranks before evaluation
            if world_size > 1:
                torch.distributed.barrier()

            if rank == 0:
                current_epoch = global_step / steps_per_epoch
                print(f"\n{'='*60}")
                print(f"Generating and evaluating molecules at step {global_step} (epoch {current_epoch:.1f})...")
                print(f"{'='*60}")

                try:
                    # Get model state dict (unwrap DDP and compiled model if needed)
                    if world_size > 1:
                        model_state_dict = model.module.state_dict()
                    else:
                        model_state_dict = model.state_dict()

                    # Clean up compiled model prefixes if present
                    if any(k.startswith('_orig_mod.') for k in model_state_dict.keys()):
                        cleaned_state_dict = {}
                        prefix = '_orig_mod.'
                        for k, v in model_state_dict.items():
                            if k.startswith(prefix):
                                cleaned_state_dict[k.replace(prefix, '', 1)] = v
                            else:
                                cleaned_state_dict[k] = v
                        model_state_dict = cleaned_state_dict

                    # Create a fresh model for generation (not DDP wrapped)
                    eval_model = EmbeddingConditionalLlamaDecoder(decoder_config)
                    eval_model.load_state_dict(model_state_dict)
                    eval_model = eval_model.to(device)
                    eval_model.eval()

                    generation_stats = generate_and_eval_with_classifier(
                        model=eval_model,
                        tokenizer=tokenizer,
                        reactions=reactions,
                        classifier_path=args.classifier_path,
                        train_smiles_set=train_smiles_set,
                        train_fingerprints=train_fingerprints,
                        num_molecules=args.num_generate,
                        temperature=args.generation_temperature,
                        top_p=args.generation_top_p,
                        max_tokens=args.max_len,
                        device=device,
                        step=global_step,
                        classifier_batch_size=args.classifier_batch_size,
                    )

                    print(f"Generation complete: {generation_stats['num_valid']}/{args.num_generate} valid molecules")
                    print(f"Validity rate: {generation_stats['validity_rate']*100:.1f}%")
                    print(f"Classifier scores: mean={generation_stats['mean_score']:.4f}, p90={generation_stats['p90_score']:.4f}, p99={generation_stats['p99_score']:.4f}")
                    print(f"Memorization: {generation_stats['frac_memorized']*100:.1f}% have Tanimoto > 0.95 to training set")

                    # Clean up eval model
                    del eval_model
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Error during molecule generation: {e}")
                    import traceback
                    traceback.print_exc()

                print(f"{'='*60}\n")

            # Synchronize all ranks after evaluation
            if world_size > 1:
                torch.distributed.barrier()

        # Save at end of each epoch
        if steps_per_epoch > 0 and global_step % steps_per_epoch == 0 and global_step > 0 and rank == 0 and args.experiment != '':
            current_epoch = global_step // steps_per_epoch
            # Get model state dict (no optimizer states)
            if world_size > 1:
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()

            # Clean up compiled model prefixes if present
            if any(k.startswith('_orig_mod.') for k in model_state_dict.keys()):
                print("Cleaning compiled model state dict before saving...")
                cleaned_state_dict = {}
                prefix = '_orig_mod.'
                for k, v in model_state_dict.items():
                    if k.startswith(prefix):
                        cleaned_state_dict[k.replace(prefix, '', 1)] = v  # Remove prefix only once from start
                    else:
                        cleaned_state_dict[k] = v
                model_state_dict = cleaned_state_dict

            save_path = f"s3://shvaibackups/robosean/{args.experiment}/epoch_{current_epoch}.pt"

            with S3Checkpoint("us-west-2").writer(save_path) as writer:
                torch.save(model_state_dict, writer)

            print(f"Model saved to: {save_path}")

        # Synchronize before next step
        if world_size > 1:
            dist.barrier()

        global_step += 1

    cleanup_ddp()

if __name__ == "__main__":
    main()