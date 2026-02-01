import os
import argparse
import torch
import torch.distributed as dist
import random
import numpy as np
import time
import glob
from utils.models import EmbeddingConditionalLlamaDecoder, PharmocophoreEncoder, create_custom_llama_config
from synthesis.dataloader import PharmacophoreDataset, get_pharmacophore, PHARMACOPHORE_TYPE_TO_ID
from synthesis.helpers import get_reactions

from s3torchconnector import S3Checkpoint

from datetime import timedelta
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

def align_pharmacophores(coords1, coords2, type_ids1, type_ids2):
    """
    Align two pharmacophores and compute RMSD.
    Uses Hungarian algorithm to match pharmacophore points by type, then computes RMSD.

    Args:
        coords1: [N, 3] target coordinates
        coords2: [M, 3] generated coordinates
        type_ids1: [N] target type IDs (0=pad)
        type_ids2: [M] generated type IDs (0=pad)

    Returns:
        rmsd: Root mean square deviation after alignment (or None if alignment fails)
        num_matched: Number of matched pharmacophore points
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    # Filter out padding (type_id == 0)
    mask1 = type_ids1 > 0
    mask2 = type_ids2 > 0

    c1 = coords1[mask1]
    c2 = coords2[mask2]
    t1 = type_ids1[mask1]
    t2 = type_ids2[mask2]

    if len(c1) == 0 or len(c2) == 0:
        return None, 0

    # Build cost matrix: distance + type penalty
    # Points of same type get distance, different types get large penalty
    cost_matrix = np.zeros((len(c1), len(c2)))
    for i in range(len(c1)):
        for j in range(len(c2)):
            if t1[i] == t2[j]:
                cost_matrix[i, j] = np.linalg.norm(c1[i] - c2[j])
            else:
                cost_matrix[i, j] = 1000.0  # Large penalty for type mismatch

    # Hungarian algorithm for optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Filter matches by type (only keep same-type matches)
    matched_distances = []
    for i, j in zip(row_ind, col_ind):
        if t1[i] == t2[j]:
            matched_distances.append(cost_matrix[i, j] ** 2)

    if len(matched_distances) == 0:
        return None, 0

    rmsd = np.sqrt(np.mean(matched_distances))
    return rmsd, len(matched_distances)


def pharmacophore_eval(
    decoder_model,
    pharm_encoder,
    tokenizer,
    reactions,
    eval_smiles,
    pharmacophore_max_len=30,
    pharmacophore_timeout=0.5,
    temperature=1.0,
    top_p=0.95,
    max_tokens=200,
    device='cuda',
    step=0
):
    """
    Pharmacophore-conditional evaluation: given target SMILES, extract their pharmacophores,
    generate synthesis paths conditioned on those pharmacophores, and measure pharmacophore alignment.

    Args:
        decoder_model: The EmbeddingConditionalLlamaDecoder model
        pharm_encoder: The PharmocophoreEncoder model
        tokenizer: The synthesis tokenizer
        reactions: Reactions dict from get_reactions()
        eval_smiles: List of SMILES to condition on
        pharmacophore_max_len: Max pharmacophore length
        pharmacophore_timeout: Timeout for pharmacophore generation
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens to generate
        device: Device to run generation on
        step: Current training step (for logging)

    Returns:
        dict with statistics about pharmacophore-conditional generation
    """
    from tqdm import tqdm
    from rdkit import Chem
    from rdkit.Chem import Draw
    from synthesis.helpers import execute_path

    decoder_model.eval()
    pharm_encoder.eval()

    # Get special token IDs
    bos_token_id = tokenizer.convert_tokens_to_ids('[BOS]')
    eos_token_id = tokenizer.convert_tokens_to_ids('[EOS]')

    # Extract pharmacophores for eval molecules
    valid_eval_data = []
    print(f"Extracting pharmacophores for {len(eval_smiles)} eval molecules...")

    for smiles in tqdm(eval_smiles, desc="Extracting pharmacophores"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        pharm = get_pharmacophore(smiles, max_len=pharmacophore_max_len, timeout_seconds=pharmacophore_timeout)
        if pharm is None:
            continue

        valid_eval_data.append({
            'smiles': smiles,
            'pharmacophore_type_ids': torch.tensor(pharm['pharmacophore_type_ids'], dtype=torch.long),
            'pharmacophore_coords': torch.tensor(pharm['pharmacophore_coords'], dtype=torch.float32),
        })

    if not valid_eval_data:
        print("No valid eval molecules with pharmacophores!")
        return {"avg_rmsd": None, "validity_rate": 0.0, "num_valid": 0}

    num_molecules = len(valid_eval_data)
    print(f"Got pharmacophores for {num_molecules} molecules")

    # Batch pharmacophores and encode
    pharm_type_ids = torch.stack([d['pharmacophore_type_ids'] for d in valid_eval_data]).to(device)
    pharm_coords = torch.stack([d['pharmacophore_coords'] for d in valid_eval_data]).to(device)

    with torch.no_grad():
        # Encode pharmacophores
        pharm_embeddings = pharm_encoder(pharm_coords, pharm_type_ids)  # [bs, seq_len, hidden]

        # Use mean pooling over non-padding positions for conditioning
        pharm_mask = (pharm_type_ids != 0).float().unsqueeze(-1)  # [bs, seq_len, 1]
        pharm_conditioning = (pharm_embeddings * pharm_mask).sum(dim=1) / (pharm_mask.sum(dim=1) + 1e-8)  # [bs, hidden]

    # Create input: just [BOS] for each molecule (conditioning comes from embeddings)
    input_ids_list = [torch.tensor([bos_token_id], device=device) for _ in range(num_molecules)]

    print(f"\nGenerating {num_molecules} synthesis paths conditioned on pharmacophores...")

    with torch.no_grad():
        generated_sequences = decoder_model.generate(
            input_ids_list=input_ids_list,
            conditioning_embeddings=pharm_conditioning,
            max_generated_tokens=max_tokens,
            stop_token_id=eos_token_id,
            temperature=temperature,
            top_p=top_p,
        )
    print(f"Generation complete, got {len(generated_sequences)} sequences")

    # Decode, parse, execute, and compute pharmacophore alignment
    valid_results = []
    rmsd_values = []
    parse_errors = 0
    execution_errors = 0
    invalid_count = 0

    print(f"Decoding, parsing, and computing pharmacophore alignment...")
    for idx, seq in enumerate(tqdm(generated_sequences, desc="Pharmacophore eval", unit="path")):
        target_data = valid_eval_data[idx]
        target_smiles = target_data['smiles']
        target_type_ids = target_data['pharmacophore_type_ids'].numpy()
        target_coords = target_data['pharmacophore_coords'].numpy()

        try:
            # Decode full sequence
            seq_list = seq.tolist() if hasattr(seq, 'tolist') else seq
            decoded_full = tokenizer.decode(seq_list)

            # Remove spaces and special tokens
            clean_str = decoded_full.replace(' ', '')
            for tok in ['[BOS]', '[EOS]', '[PAD]']:
                clean_str = clean_str.replace(tok, '')
            clean_str = clean_str.strip()

            # Check if it looks like a valid synthesis path
            if not clean_str.startswith('<ADD>'):
                parse_errors += 1
                invalid_count += 1
                continue

            # Try to execute the synthesis path
            result, _ = execute_path(clean_str, reactions)

            if result is None:
                execution_errors += 1
                invalid_count += 1
                continue

            # Validate with RDKit
            result_mol = Chem.MolFromSmiles(result)
            if result_mol is None:
                invalid_count += 1
                continue

            # Get pharmacophore of generated molecule
            gen_pharm = get_pharmacophore(result, max_len=pharmacophore_max_len, timeout_seconds=pharmacophore_timeout)
            if gen_pharm is None:
                invalid_count += 1
                continue

            gen_type_ids = np.array(gen_pharm['pharmacophore_type_ids'])
            gen_coords = gen_pharm['pharmacophore_coords']

            # Align pharmacophores and compute RMSD
            rmsd, num_matched = align_pharmacophores(target_coords, gen_coords, target_type_ids, gen_type_ids)

            if rmsd is not None:
                rmsd_values.append(rmsd)
                valid_results.append({
                    'target': target_smiles,
                    'generated': Chem.MolToSmiles(result_mol, canonical=True),
                    'rmsd': rmsd,
                    'num_matched': num_matched,
                    'path': clean_str,
                    'target_coords': target_coords,
                    'target_types': target_type_ids,
                    'gen_coords': gen_coords,
                    'gen_types': gen_type_ids,
                })

        except Exception as e:
            print(f"Exception in pharmacophore eval: {e}")
            invalid_count += 1
            continue

    num_valid = len(valid_results)
    validity_rate = num_valid / num_molecules if num_molecules > 0 else 0.0
    avg_rmsd = np.mean(rmsd_values) if rmsd_values else None

    print(f"\nPharmacophore eval results:")
    print(f"  Valid: {num_valid}/{num_molecules} ({validity_rate*100:.1f}%)")
    print(f"  Parse errors: {parse_errors}")
    print(f"  Execution errors: {execution_errors}")
    print(f"  Avg pharmacophore RMSD: {avg_rmsd:.4f}" if avg_rmsd else "  Avg pharmacophore RMSD: N/A")

    # Log some examples
    for i, res in enumerate(valid_results[:5]):
        print(f"  Example {i+1}: target={res['target'][:30]}... -> generated={res['generated'][:30]}... (RMSD={res['rmsd']:.3f}, matched={res['num_matched']})")

    # Log to wandb
    if num_valid > 0:
        log_dict = {
            "pharm_eval/validity_rate": validity_rate,
            "pharm_eval/num_valid": num_valid,
            "pharm_eval/parse_errors": parse_errors,
            "pharm_eval/execution_errors": execution_errors,
        }

        if avg_rmsd is not None:
            log_dict["pharm_eval/avg_rmsd"] = avg_rmsd
            log_dict["pharm_eval/rmsd_histogram"] = wandb.Histogram(rmsd_values)

        # Log images of target vs generated for first few valid
        mol_images = []
        pharm_images = []

        # Pharmacophore type colors
        PHARM_COLORS = {
            1: 'blue',      # Donor
            2: 'red',       # Acceptor
            3: 'orange',    # Aromatic
            4: 'green',     # Hydrophobe
            5: 'lightgreen',# LumpedHydrophobe
            6: 'purple',    # PosIonizable
            7: 'brown',     # NegIonizable
        }
        PHARM_NAMES = {
            1: 'Donor', 2: 'Acceptor', 3: 'Aromatic', 4: 'Hydrophobe',
            5: 'LumpedHydrophobe', 6: 'PosIonizable', 7: 'NegIonizable'
        }

        for res in valid_results[:10]:
            try:
                # 2D molecule comparison
                target_mol = Chem.MolFromSmiles(res['target'])
                gen_mol = Chem.MolFromSmiles(res['generated'])
                if target_mol and gen_mol:
                    img = Draw.MolsToGridImage([target_mol, gen_mol], molsPerRow=2,
                                               subImgSize=(200, 200),
                                               legends=[f"Target", f"Generated (RMSD={res['rmsd']:.2f})"])
                    mol_images.append(wandb.Image(np.array(img), caption=f"RMSD: {res['rmsd']:.3f}"))

                # 3D pharmacophore comparison - overlaid after Kabsch alignment
                import matplotlib.pyplot as plt
                from scipy.optimize import linear_sum_assignment
                import io
                from PIL import Image as PILImage

                target_coords = res['target_coords']
                target_types = res['target_types']
                gen_coords = res['gen_coords']
                gen_types = res['gen_types']

                # Filter out padding
                t_mask = target_types > 0
                g_mask = gen_types > 0
                t_coords = target_coords[t_mask]
                t_types = target_types[t_mask]
                g_coords = gen_coords[g_mask]
                g_types = gen_types[g_mask]

                if len(t_coords) >= 3 and len(g_coords) >= 3:
                    # Find matched pairs using Hungarian algorithm (same as align_pharmacophores)
                    n1, n2 = len(t_coords), len(g_coords)
                    cost_matrix = np.zeros((n1, n2))
                    for ii in range(n1):
                        for jj in range(n2):
                            dist = np.linalg.norm(t_coords[ii] - g_coords[jj])
                            type_penalty = 0 if t_types[ii] == g_types[jj] else 1000.0
                            cost_matrix[ii, jj] = dist + type_penalty
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)

                    # Get matched pairs (same type only)
                    t_matched, g_matched = [], []
                    for ii, jj in zip(row_ind, col_ind):
                        if t_types[ii] == g_types[jj]:
                            t_matched.append(t_coords[ii])
                            g_matched.append(g_coords[jj])

                    if len(t_matched) >= 3:
                        t_matched = np.array(t_matched)
                        g_matched = np.array(g_matched)

                        # Kabsch alignment on matched pairs
                        t_centered = t_matched - t_matched.mean(axis=0)
                        g_centered = g_matched - g_matched.mean(axis=0)
                        H = g_centered.T @ t_centered
                        U, S, Vt = np.linalg.svd(H)
                        R = Vt.T @ U.T
                        if np.linalg.det(R) < 0:
                            Vt[-1, :] *= -1
                            R = Vt.T @ U.T

                        # Apply rotation to ALL generated coords (not just matched)
                        g_all_centered = g_coords - g_matched.mean(axis=0)
                        g_aligned = (g_all_centered @ R) + t_matched.mean(axis=0)

                        fig = plt.figure(figsize=(6, 6))
                        ax = fig.add_subplot(111, projection='3d')

                        # Target (circles)
                        for coord, tid in zip(t_coords, t_types):
                            ax.scatter(*coord, c=PHARM_COLORS.get(tid, 'gray'), s=150, marker='o', alpha=0.7, edgecolors='black', linewidths=1)

                        # Generated aligned (X markers)
                        for coord, tid in zip(g_aligned, g_types):
                            ax.scatter(*coord, c=PHARM_COLORS.get(tid, 'gray'), s=150, marker='x', alpha=0.9, linewidths=2)

                        ax.set_title(f'Target (o) vs Generated (x) | RMSD={res["rmsd"]:.2f}')
                        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

                        # Set equal aspect
                        all_coords = np.vstack([t_coords, g_aligned])
                        max_range = np.max(np.abs(all_coords - all_coords.mean(axis=0))) * 1.3
                        mid = all_coords.mean(axis=0)
                        ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
                        ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
                        ax.set_zlim(mid[2]-max_range, mid[2]+max_range)

                        plt.tight_layout()

                        # Save to buffer (avoids tostring_rgb deprecation)
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=100)
                        buf.seek(0)
                        pharm_img = np.array(PILImage.open(buf))
                        pharm_images.append(wandb.Image(pharm_img, caption=f"RMSD: {res['rmsd']:.3f}"))
                        buf.close()
                        plt.close(fig)

            except Exception as e:
                print(f"Error creating comparison image: {e}")

        if mol_images:
            log_dict["pharm_eval/mol_comparisons"] = mol_images
        if pharm_images:
            log_dict["pharm_eval/pharmacophore_3d"] = pharm_images

        wandb.log(log_dict, step=step)

    decoder_model.train()
    pharm_encoder.train()

    return {
        "avg_rmsd": avg_rmsd,
        "validity_rate": validity_rate,
        "num_valid": num_valid,
        "num_molecules": num_molecules,
        "parse_errors": parse_errors,
        "execution_errors": execution_errors,
        "results": valid_results,
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
    ## ARGS GO HERE
    parser = argparse.ArgumentParser(description="Train MolGen2")

    parser.add_argument("--batch_size", type=int, default=150)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--checkpoint", type=str, default='s3://shvaibackups/robosean/500m_robosean_molprefix2/540001.pt')
    parser.add_argument("--compile", action="store_true", help="Compile model with torch.compile")

    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=1_000_000)
    parser.add_argument("--max_grad_norm", type=float, default=1.5)

    parser.add_argument("--max_len", type=int, default=200, help="Input len")

    parser.add_argument("--save_every_k_steps", type=int, default=15_000)
    parser.add_argument("--generate_every_k_steps", type=int, default=5_000, help="Generate and log molecules every K steps")
    parser.add_argument("--num_generate", type=int, default=100, help="Number of molecules to generate for validation")
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

    decoder_model = EmbeddingConditionalLlamaDecoder(decoder_config)

    # Create pharmacophore encoder config and model
    # 8 pharmacophore types (0=pad, 1-7 = types)
    pharm_vocab_size = len(PHARMACOPHORE_TYPE_TO_ID) + 1  # +1 for padding token (0)
    pharm_encoder_config = create_custom_llama_config(
        vocab_size=pharm_vocab_size,
        hidden_size=args.emb_dim,
        intermediate_size=int(args.emb_dim * args.intermediate_size_multiplier),
        num_hidden_layers=4,  # Smaller encoder for pharmacophores
        num_attention_heads=args.num_attention_heads,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=0,
        cls_token_id=0,
        max_position_embeddings=30,  # pharmacophore max len
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        attention_dropout=0.0,
        rope_base=10_000,
        use_rope=False,  # No positional encoding for pharmacophores (permutation invariant)
    )

    pharm_encoder = PharmocophoreEncoder(pharm_encoder_config)

    # Print model info on rank 0
    if rank == 0:
        from utils.models import count_parameters
        decoder_params = count_parameters(decoder_model)
        pharm_params = count_parameters(pharm_encoder)
        print(f"\nDecoder Configuration:")
        print(f"  Vocab size: {decoder_config.vocab_size}")
        print(f"  Hidden size: {decoder_config.hidden_size}")
        print(f"  Intermediate size: {decoder_config.intermediate_size}")
        print(f"  Num layers: {decoder_config.num_hidden_layers}")
        print(f"  Num attention heads: {decoder_config.num_attention_heads}")
        print(f"  Max position embeddings: {decoder_config.max_position_embeddings}")
        print(f"  Decoder parameters: {decoder_params:,} ({decoder_params/1e6:.1f}M)")
        print(f"\nPharmacophore Encoder:")
        print(f"  Vocab size: {pharm_encoder_config.vocab_size}")
        print(f"  Num layers: {pharm_encoder_config.num_hidden_layers}")
        print(f"  Parameters: {pharm_params:,} ({pharm_params/1e6:.1f}M)")
        print(f"\nTotal parameters: {decoder_params + pharm_params:,} ({(decoder_params + pharm_params)/1e6:.1f}M)")
        print()

    # Move models to device
    decoder_model = decoder_model.to(device)
    pharm_encoder = pharm_encoder.to(device)

    if args.checkpoint != '':
        print(f"Loading decoder from checkpoint: {args.checkpoint}")

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
                    new_checkpoint[k.replace(prefix, '', 1)] = v
                else:
                    new_checkpoint[k] = v
            del checkpoint
            checkpoint = new_checkpoint

        decoder_model.load_state_dict(checkpoint)
        del checkpoint
        torch.cuda.empty_cache()

        if rank == 0:
            print("Decoder checkpoint loaded and memory cleared")

    if args.compile:
        decoder_model = torch.compile(decoder_model)
        pharm_encoder = torch.compile(pharm_encoder)

    # Wrap with DDP
    if world_size > 1:
        decoder_model = DDP(decoder_model, device_ids=[local_rank])
        pharm_encoder = DDP(pharm_encoder, device_ids=[local_rank])

    # Initialize optimizer for both models
    all_params = list(decoder_model.parameters()) + list(pharm_encoder.parameters())
    optimizer = AdamW(all_params, lr=args.max_lr)

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
    decoder_model.train()
    pharm_encoder.train()
    global_step = 0
    total_start_time = time.time()

    # Initialize wandb only on rank 0
    if rank == 0:
        wandb.init(
            project="robosean_pharmacophore",
            name=args.experiment,
            config=args,
        )

    # Load eval SMILES once at startup (only on rank 0)
    eval_smiles_pool = []
    if rank == 0:
        import pandas as pd
        try:
            zinc_path = "s3://shvaibackups/unibio_data/zinc22/shuffled/0000.parquet"
            df = pd.read_parquet(zinc_path, columns=['smiles'])
            eval_smiles_pool = df['smiles'].sample(n=min(1000, len(df))).tolist()
            print(f"Loaded {len(eval_smiles_pool)} eval SMILES into memory")
        except Exception as e:
            print(f"Error loading eval SMILES: {e}")

    # Get all .parquet files from S3
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

    # Create dataset and dataloader with pharmacophore support
    train_dataset = PharmacophoreDataset(
        file_list=mol_files,
        tokenizer=tokenizer,
        max_len=args.max_len,
        pharmacophore_max_len=30,
        pharmacophore_timeout=0.5,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    train_iterator = iter(train_dataloader)

    while global_step < args.max_steps:
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
            pharm_type_ids = batch['pharmacophore_type_ids'].to(device)
            pharm_coords = batch['pharmacophore_coords'].to(device)

            with torch.autocast('cuda', dtype=torch.bfloat16):
                # Encode pharmacophores
                pharm_embeddings = pharm_encoder(pharm_coords, pharm_type_ids)  # [bs, seq_len, hidden]

                # Mean pool over non-padding positions for conditioning
                pharm_mask = (pharm_type_ids != 0).float().unsqueeze(-1)  # [bs, seq_len, 1]
                pharm_conditioning = (pharm_embeddings * pharm_mask).sum(dim=1) / (pharm_mask.sum(dim=1) + 1e-8)  # [bs, hidden]

                # For causal language modeling, labels are the same as input_ids
                # The model will shift them internally for next-token prediction
                outputs = decoder_model(
                    input_ids=token_ids,
                    attention_mask=None,
                    conditioning_embeddings=pharm_conditioning,
                    labels=token_ids,
                )

                step_loss = outputs['loss'] / args.grad_accum_steps

            total_loss = total_loss + step_loss
            step_loss.backward()

        # Synchronize gradients across processes
        if world_size > 1:
            dist.barrier()

        # Clip gradients and step
        all_params = list(decoder_model.parameters()) + list(pharm_encoder.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=args.max_grad_norm)
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

        # Pharmacophore-conditional evaluation
        if (global_step % args.generate_every_k_steps == 0 and global_step > 0) or global_step == 10:
            # Synchronize all ranks before evaluation
            if world_size > 1:
                torch.distributed.barrier()

            if rank == 0:
                print(f"\n{'='*60}")
                print(f"Running pharmacophore-conditional evaluation at step {global_step}...")
                print(f"{'='*60}")

                try:
                    # Get model state dicts (unwrap DDP and compiled model if needed)
                    if world_size > 1:
                        decoder_state_dict = decoder_model.module.state_dict()
                        pharm_state_dict = pharm_encoder.module.state_dict()
                    else:
                        decoder_state_dict = decoder_model.state_dict()
                        pharm_state_dict = pharm_encoder.state_dict()

                    # Clean up compiled model prefixes if present
                    def clean_state_dict(sd):
                        if any(k.startswith('_orig_mod.') for k in sd.keys()):
                            cleaned = {}
                            for k, v in sd.items():
                                if k.startswith('_orig_mod.'):
                                    cleaned[k.replace('_orig_mod.', '', 1)] = v
                                else:
                                    cleaned[k] = v
                            return cleaned
                        return sd

                    decoder_state_dict = clean_state_dict(decoder_state_dict)
                    pharm_state_dict = clean_state_dict(pharm_state_dict)

                    # Create fresh models for evaluation (not DDP wrapped)
                    eval_decoder = EmbeddingConditionalLlamaDecoder(decoder_config)
                    eval_decoder.load_state_dict(decoder_state_dict)
                    eval_decoder = eval_decoder.to(device)
                    eval_decoder.eval()

                    eval_pharm_encoder = PharmocophoreEncoder(pharm_encoder_config)
                    eval_pharm_encoder.load_state_dict(pharm_state_dict)
                    eval_pharm_encoder = eval_pharm_encoder.to(device)
                    eval_pharm_encoder.eval()

                    # Sample SMILES for pharmacophore eval
                    if eval_smiles_pool:
                        pharm_eval_smiles = random.sample(eval_smiles_pool, min(args.num_generate, len(eval_smiles_pool)))

                        pharm_eval_stats = pharmacophore_eval(
                            decoder_model=eval_decoder,
                            pharm_encoder=eval_pharm_encoder,
                            tokenizer=tokenizer,
                            reactions=reactions,
                            eval_smiles=pharm_eval_smiles,
                            pharmacophore_max_len=30,
                            pharmacophore_timeout=0.5,
                            temperature=args.generation_temperature,
                            top_p=args.generation_top_p,
                            max_tokens=args.max_len,
                            device=device,
                            step=global_step
                        )

                        print(f"Pharmacophore eval: {pharm_eval_stats['num_valid']}/{pharm_eval_stats['num_molecules']} valid")
                        if pharm_eval_stats['avg_rmsd'] is not None:
                            print(f"Average pharmacophore RMSD: {pharm_eval_stats['avg_rmsd']:.4f}")

                    # Clean up eval models
                    del eval_decoder, eval_pharm_encoder
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Error during pharmacophore evaluation: {e}")
                    import traceback
                    traceback.print_exc()

                print(f"{'='*60}\n")

            # Synchronize all ranks after evaluation
            if world_size > 1:
                torch.distributed.barrier()

        # Saving
        if global_step % args.save_every_k_steps == 1 and rank == 0 and args.experiment != '':
            # Get model state dicts (no optimizer states)
            if world_size > 1:
                decoder_state_dict = decoder_model.module.state_dict()
                pharm_state_dict = pharm_encoder.module.state_dict()
            else:
                decoder_state_dict = decoder_model.state_dict()
                pharm_state_dict = pharm_encoder.state_dict()

            # Clean up compiled model prefixes if present
            def clean_state_dict(sd):
                if any(k.startswith('_orig_mod.') for k in sd.keys()):
                    print("Cleaning compiled model state dict before saving...")
                    cleaned = {}
                    for k, v in sd.items():
                        if k.startswith('_orig_mod.'):
                            cleaned[k.replace('_orig_mod.', '', 1)] = v
                        else:
                            cleaned[k] = v
                    return cleaned
                return sd

            decoder_state_dict = clean_state_dict(decoder_state_dict)
            pharm_state_dict = clean_state_dict(pharm_state_dict)

            # Save decoder
            decoder_save_path = f"s3://shvaibackups/robosean_pharmacophore/{args.experiment}/decoder_{global_step}.pt"
            with S3Checkpoint("us-west-2").writer(decoder_save_path) as writer:
                torch.save(decoder_state_dict, writer)
            print(f"Decoder saved to: {decoder_save_path}")

            # Save pharmacophore encoder
            pharm_save_path = f"s3://shvaibackups/robosean_pharmacophore/{args.experiment}/pharm_encoder_{global_step}.pt"
            with S3Checkpoint("us-west-2").writer(pharm_save_path) as writer:
                torch.save(pharm_state_dict, writer)
            print(f"Pharmacophore encoder saved to: {pharm_save_path}")

        # Synchronize before next step
        if world_size > 1:
            dist.barrier()

        global_step += 1

    cleanup_ddp()

if __name__ == "__main__":
    main()