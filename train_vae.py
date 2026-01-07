import os
import argparse
import torch
import torch.distributed as dist
import random
import numpy as np
import pandas as pd
import time
import glob
from pathlib import Path
from utils.models import Smile2SmileVAE, LlamaConfig
from utils.dataloader import SmilesVAEDataset
from tokenizers import Tokenizer

from s3torchconnector import S3Checkpoint

from datetime import timedelta
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def generation_eval(list_of_smiles, vae, tokenizer, max_len, noise_ratio=1.0):
    """
    Encode/decode SMILES and compute reconstruction quality.
    """
    device = next(vae.parameters()).device
    valid_smiles = []
    original_mols = []

    pad_token_id = tokenizer.token_to_id("[PAD]")

    for s in list_of_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            canon_smiles = Chem.MolToSmiles(mol, canonical=True)
            valid_smiles.append(canon_smiles)
            original_mols.append(mol)

    if len(valid_smiles) == 0:
        return 0.0, 0.0

    # Tokenize all valid SMILES
    all_ids = []
    all_masks = []
    for smiles in valid_smiles:
        ids = tokenizer.encode(f'[BOS]{smiles}[EOS]').ids
        seq_len = len(ids)
        if seq_len > max_len:
            ids = ids[:max_len]
            seq_len = max_len
        padding_len = max_len - seq_len
        padded_ids = ids + [pad_token_id] * padding_len
        attention_mask = [1] * seq_len + [0] * padding_len
        all_ids.append(padded_ids)
        all_masks.append(attention_mask)

    input_ids = torch.tensor(all_ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(all_masks, dtype=torch.bool, device=device)

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            encode_out = vae.encode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                noise_ratio=noise_ratio
            )
            latents = encode_out['latent']
            generated_ids = vae.decode(latent=latents, temperature=0.0)

    # Decode generated tokens to SMILES and compute similarity
    similarities = []
    valid_count = 0
    for i, (gen_ids, original_mol) in enumerate(zip(generated_ids, original_mols)):
        # Decode tokens to SMILES
        tokens = [tokenizer.id_to_token(int(tid)) for tid in gen_ids if tid != pad_token_id]
        # Remove special tokens and join
        smiles_tokens = []
        for t in tokens:
            if t in ['[BOS]', '[EOS]', '[PAD]', '[CLS]', '[MASK]', '[UNK]']:
                continue
            smiles_tokens.append(t)
        new_smiles = ''.join(smiles_tokens)

        new_mol = Chem.MolFromSmiles(new_smiles)
        if new_mol is not None:
            valid_count += 1
            fp1 = AllChem.GetMorganFingerprintAsBitVect(original_mol, radius=3, nBits=1024)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(new_mol, radius=3, nBits=1024)
            sim = DataStructs.TanimotoSimilarity(fp1, fp2)
            similarities.append(sim)

    avg_sim = np.mean(similarities) if len(similarities) > 0 else 0.0
    validity_rate = valid_count / len(valid_smiles) if len(valid_smiles) > 0 else 0.0

    return avg_sim, validity_rate

def do_evals_in_train_loop(smiles_to_use, vae, tokenizer, max_len, device):
    """Run evaluation at different noise levels."""
    vae.eval()

    results = {}
    for noise in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        print(f"  Evaluating with noise_ratio={noise}")
        sim, valid = generation_eval(
            smiles_to_use,
            vae=vae,
            tokenizer=tokenizer,
            max_len=max_len,
            noise_ratio=noise,
        )
        results[f'eval_valid_noise_{noise}'] = valid
        results[f'eval_sim_noise_{noise}'] = sim
        print(f"    Validity: {valid*100:.1f}%, Similarity: {sim:.3f}")

    vae.train()
    return results


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
    ## ARGS GO HERE
    parser = argparse.ArgumentParser(description="Train VAE")

    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--checkpoint", type=str, default='')
    parser.add_argument("--compile", action="store_true", help="Compile model with torch.compile")

    parser.add_argument("--max_lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=2e-4)
    parser.add_argument("--max_steps", type=int, default=1_000_000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--max_mol_size", type=int, default=120, help="Maximum molecule size (number of atoms)")

    parser.add_argument("--save_every_k_steps", type=int, default=50_000)
    parser.add_argument("--generate_every_k_steps", type=int, default=2_500, help="Generate and log molecules every K steps")
    parser.add_argument("--num_generate", type=int, default=100, help="Number of molecules to generate for validation")
    parser.add_argument("--generation_temperature", type=float, default=1.0, help="Temperature for molecule generation")
    parser.add_argument("--generation_top_p", type=float, default=0.9, help="Top-p for molecule generation")

    # losses
    parser.add_argument("--kl_loss_scale", type=float, default=1e-4, help="KL loss scale")
    parser.add_argument("--contrastive_loss_scale", type=float, default=1.0, help="KL loss scale")

    ## 100m model: (emb 512, layers 24, heads 8, intermediate_size_multiplier 4)
    ## 200m model: (emb 768, layers 28, heads 12, intermediate_size_multiplier 3)
    ## 300m model: (emb 768, layers 32, heads 12, intermediate_size_multiplier 4)
    ## 500m model: (emb 1024, layers 36, heads 16, intermediate_size_multiplier 3)
    ## esmc 600m model: emb 1152, heads 18, layers 36, intermediate_size_multiplier 8/3
    ## 1b model: (emb 1280, layers 40, heads 20, intermediate_size_multiplier 4)
    ## 1.8b model: (emb 1536, layers 48, heads 24, intermediate_size_multiplier 4)
    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--intermediate_size_multiplier", type=float, default=4)
    parser.add_argument("--num_layers", type=int, default=28)
    parser.add_argument("--num_attention_heads", type=int, default=12)

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

    # Load tokenizer
    tokenizer = Tokenizer.from_file('tokenizers/smiles_tokenizer_simple/tokenizer.json')

    # Create model config for molecules
    vocab_size = max(tokenizer.get_vocab().values()) + 1

    encoder_config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=args.emb_dim,
        intermediate_size=int(args.emb_dim * args.intermediate_size_multiplier),
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        pad_token_id=tokenizer.token_to_id('[PAD]'),
        bos_token_id=tokenizer.token_to_id('[BOS]'),
        eos_token_id=tokenizer.token_to_id('[EOS]'),
        cls_token_id=tokenizer.token_to_id('[BOS]'),  # Use BOS as CLS
        max_position_embeddings=args.max_mol_size,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        attention_dropout=0.0,
        rope_base=10_000,
        use_rope=True,
    )

    decoder_config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=args.emb_dim,
        intermediate_size=int(args.emb_dim * args.intermediate_size_multiplier),
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        pad_token_id=tokenizer.token_to_id('[PAD]'),
        bos_token_id=tokenizer.token_to_id('[BOS]'),
        eos_token_id=tokenizer.token_to_id('[EOS]'),
        cls_token_id=tokenizer.token_to_id('[BOS]'),  # Use BOS as CLS
        max_position_embeddings=args.max_mol_size,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        attention_dropout=0.0,
        rope_base=10_000,
        use_rope=True,
    )

    model = Smile2SmileVAE(encoder_config, decoder_config)

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
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=args.max_lr)

    # Create LR scheduler with warmup and linear cooldown
    warmup_steps = 20
    cooldown_steps = args.max_steps - warmup_steps

    # Warmup scheduler: goes from very small LR to max_lr over warmup_steps
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)

    # Cooldown scheduler: goes from max_lr to min_lr over remaining steps
    cooldown_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=args.min_lr/args.max_lr, total_iters=cooldown_steps)

    # Combine schedulers
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cooldown_scheduler], milestones=[warmup_steps])

    # Training loop
    model.train()
    global_step = 0
    total_start_time = time.time()

    # Initialize wandb only on rank 0
    if rank == 0:
        wandb.init(
            project="mol_vae_v9000",
            name=args.experiment,
            config=args,
        )

    # Find molecule data files
    #if os.path.exists("/home/ubuntu/shv-storage/unibio_data/zinc22/processed/"):
    #mol_data_path = "/home/ubuntu/zinc22_shuffled/"
    #mol_data_path = '/home/ubuntu/shv-storage/unibio_data/zinc22/processed/'

    #mol_files = [f for f in glob.glob(os.path.join(mol_data_path, "*.parquet"))]
    #print(f"Found {len(mol_files)} Zinc22 files in {mol_data_path}.")

    #mol_data_path = 's3://shvaibackups/unibio_data/zinc22/shuffled/'

    # # Get all .parquet files from S3
    eval_file = 's3://shvaibackups/unibio_data/zinc22/shuffled/0000.parquet'
    import boto3
    s3_client = boto3.client('s3', region_name='us-west-2')
    bucket_name = 'shvaibackups'
    prefix = 'unibio_data/zinc22/shuffled/'

    mol_files = []
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith('.parquet'):
                    mol_files.append(f"s3://{bucket_name}/{obj['Key']}")
    if rank == 0:
       print(f"Found {len(mol_files)} molecule parquet files.")

    mol_files = [m for m in mol_files if m != eval_file]
    # Create dataset and dataloader
    train_dataset = SmilesVAEDataset(
        file_list=mol_files,
        tokenizer=tokenizer,
        max_len=args.max_mol_size,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    train_iterator = iter(train_dataloader)

    # Load eval SMILES for generation evaluation
    eval_smiles = None
    if rank == 0:
        print("Loading eval SMILES from S3...")
        eval_df = pd.read_parquet(eval_file)
        eval_smiles = eval_df['smiles'].sample(n=256, random_state=42).tolist()
        print(f"Loaded {len(eval_smiles)} eval SMILES")

    while global_step < args.max_steps:
        time_start = time.time()
        optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=device)
        total_decoder_loss = torch.tensor(0.0, device=device)
        total_kl_loss = torch.tensor(0.0, device=device)
        total_contrastive_loss = torch.tensor(0.0, device=device)
        mu_norm = 0.0
        std_latents = 0.0

        for grad_accum_step in range(args.grad_accum_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                batch = next(train_iterator)

            canonical_ids = batch['canonical_ids'].to(device)
            canonical_attention_mask = batch['canonical_attention_mask'].to(device)
            random_ids = batch['random_ids'].to(device)
            random_attention_mask = batch['random_attention_mask'].to(device)

            with torch.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(
                    canonical_input_ids=canonical_ids,
                    canonical_attention_mask=canonical_attention_mask,
                    random_view_input_ids=random_ids,
                    random_view_attention_mask=random_attention_mask,
                )

                decoder_loss = outputs['decoder_loss']
                kl_loss = outputs['kl_loss']
                contrastive_loss = outputs['contrastive_loss']

                step_loss = decoder_loss + kl_loss * args.kl_loss_scale + contrastive_loss * args.contrastive_loss_scale
                step_loss = step_loss / args.grad_accum_steps

            total_loss = total_loss + step_loss.detach()
            total_decoder_loss = total_decoder_loss + decoder_loss.detach() / args.grad_accum_steps
            total_kl_loss = total_kl_loss + kl_loss.detach() / args.grad_accum_steps
            total_contrastive_loss = total_contrastive_loss + contrastive_loss.detach() / args.grad_accum_steps
            mu_norm += outputs['mu_norm'] / args.grad_accum_steps
            std_latents += outputs['std_latents'] / args.grad_accum_steps

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

            log_str = (
                f"Step {global_step} | "
                f"Time: {time_taken:.2f}s | "
                f"Loss: {total_loss.item():.4f} | "
                f"Dec: {total_decoder_loss.item():.4f} | "
                f"KL: {total_kl_loss.item():.4f} | "
                f"Contr: {total_contrastive_loss.item():.4f} | "
                f"GradNorm: {grad_norm.item():.2f} | "
                f"LR: {current_lr:.2e} | "
                f"μ-norm: {mu_norm:.2f} | "
                f"σ: {std_latents:.2f} | "
                f"Data: {data_done:.2f}M"
            )
            print(log_str)

            log_args = {
                "train_loss": total_loss.item(),
                "decoder_loss": total_decoder_loss.item(),
                "kl_loss": total_kl_loss.item(),
                "contrastive_loss": total_contrastive_loss.item(),
                "lr": current_lr,
                "step_time": time_taken,
                "grad_norm": grad_norm.item(),
                "data_seen_millions": data_done,
                "mu_norm": mu_norm,
                "std_latents": std_latents,
            }

            wandb.log(log_args, step=global_step)

        # Generation evaluation (encode/decode)
        if (global_step % args.generate_every_k_steps == 0 and global_step > 0) or global_step == 10:
            # Synchronize all ranks before evaluation
            if world_size > 1:
                torch.distributed.barrier()

            if rank == 0 and eval_smiles is not None:
                print(f"\n{'='*60}")
                print(f"Running encode/decode evaluation at step {global_step}...")
                print(f"{'='*60}")

                try:
                    # Get the model for eval (unwrap DDP if needed)
                    if world_size > 1:
                        eval_model = model.module
                    else:
                        eval_model = model

                    eval_model.eval()

                    # Run evaluation at different noise levels
                    eval_results = do_evals_in_train_loop(
                        smiles_to_use=eval_smiles,
                        vae=eval_model,
                        tokenizer=tokenizer,
                        max_len=args.max_mol_size,
                        device=device,
                    )

                    # Log results
                    print(f"Evaluation results:")
                    for k, v in eval_results.items():
                        print(f"  {k}: {v:.4f}")

                    wandb.log(eval_results, step=global_step)

                    eval_model.train()

                except Exception as e:
                    print(f"Error during generation eval: {e}")
                    import traceback
                    traceback.print_exc()

                print(f"{'='*60}\n")

            # Synchronize all ranks after evaluation
            if world_size > 1:
                torch.distributed.barrier()

        # Saving
        if global_step % args.save_every_k_steps == 1 and rank == 0 and args.experiment != '':# and global_step > args.save_every_k_steps:
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

            save_path = f"s3://shvaibackups/mol_vae_v9000/{args.experiment}/{global_step}.pt"

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