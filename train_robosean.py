import os
import argparse
import torch
import torch.distributed as dist
import random
import numpy as np
import time
import glob
from utils.models import EmbeddingConditionalLlamaDecoder, create_custom_llama_config
from synthesis.dataloader import SynthesisPathDataset
from synthesis.helpers import get_reactions

from s3torchconnector import S3Checkpoint

from datetime import timedelta
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

def generate_and_log_molecules(
    model,
    tokenizer,
    reactions,
    num_molecules=100,
    temperature=1.0,
    top_p=0.95,
    max_tokens=200,
    device='cuda',
    step=0
):
    """
    Generate synthesis paths using the model, execute them, and log valid molecules to wandb.

    Args:
        model: The EmbeddingConditionalLlamaDecoder model
        tokenizer: The synthesis tokenizer
        reactions: Reactions dict from get_reactions()
        num_molecules: Number of molecules to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens to generate
        device: Device to run generation on
        step: Current training step (for logging)

    Returns:
        dict with statistics about generation
    """
    import re
    import pandas as pd
    from tqdm import tqdm
    from rdkit import Chem
    from rdkit.Chem import Draw
    from synthesis.dataloader import property_tokens, calculate_properties
    from synthesis.helpers import execute_path

    model.eval()

    # Get special token IDs
    bos_token_id = tokenizer.convert_tokens_to_ids('[BOS]')
    eos_token_id = tokenizer.convert_tokens_to_ids('[EOS]')

    # Get all property token IDs
    property_token_ids = {}
    for token in property_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.convert_tokens_to_ids('[UNK]'):
            property_token_ids[token] = token_id

    # Load some ZINC molecules to get realistic property distributions
    zinc_properties = []
    zinc_path = "s3://shvaibackups/unibio_data/zinc22/shuffled/0000.parquet"

    try:
        df = pd.read_parquet(zinc_path, columns=['smiles'])
        zinc_smiles_examples = df['smiles'].sample(n=100).tolist()
        for smiles in zinc_smiles_examples[:50]:
            try:
                props = calculate_properties(smiles)
                zinc_properties.append(props['quantized_properties'])
            except:
                pass
        print(f"Loaded properties from {len(zinc_properties)} ZINC molecules")
    except Exception as e:
        print(f"Error loading ZINC file: {e}")

    # Create initial input: property tokens + BOS for each molecule
    input_ids_list = []
    molecule_properties = []  # Track which properties each molecule was conditioned on

    for _ in range(num_molecules):
        selected_properties = []
        if random.random() < 0.5 and zinc_properties:
            template_props = random.choice(zinc_properties)
            for prop in template_props:
                if random.random() < 0.5:
                    selected_properties.append(prop)

        char_tokens = [property_token_ids[prop] for prop in selected_properties if prop in property_token_ids]
        random.shuffle(char_tokens)

        tokens = char_tokens + [bos_token_id]
        input_ids_list.append(torch.tensor(tokens, device=device))
        molecule_properties.append(selected_properties)

    print(f"\nGenerating {num_molecules} synthesis paths...")

    with torch.no_grad():
        generated_sequences = model.generate(
            input_ids_list=input_ids_list,
            conditioning_embeddings=None,
            max_generated_tokens=max_tokens,
            stop_token_id=eos_token_id,
            temperature=temperature,
            top_p=top_p,
        )
    print(f"Generation complete, got {len(generated_sequences)} sequences")

    # Decode, parse, and execute synthesis paths
    valid_molecules = []
    valid_smiles = []
    valid_paths = []
    valid_indices = []
    invalid_count = 0
    parse_errors = 0
    execution_errors = 0

    print(f"Decoding, parsing, and executing synthesis paths...")
    for idx, seq in enumerate(tqdm(generated_sequences, desc="Validating", unit="path")):
        try:
            # Decode full sequence
            print(f"[DEBUG] Got sequence: {seq}")
            # Handle both tensor and list
            seq_list = seq.tolist() if hasattr(seq, 'tolist') else seq
            decoded_full = tokenizer.decode(seq_list)

            # Debug: print all sequences
            print(f"\n[DEBUG] Sequence {idx}: {decoded_full}")

            # Remove property tokens, BOS, EOS, PAD, and spaces (tokenizer adds spaces between tokens)
            clean_str = decoded_full.replace(' ', '')  # Remove all spaces first
            for tok in ['[BOS]', '[EOS]', '[PAD]'] + property_tokens:
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
            mol = Chem.MolFromSmiles(result)
            if mol is None:
                invalid_count += 1
                continue

            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

            valid_molecules.append(mol)
            valid_smiles.append(canonical_smiles)
            valid_paths.append(clean_str)
            valid_indices.append(idx)

        except Exception as e:
            print(f"Exception in eval: {e}")
            invalid_count += 1
            continue

    num_valid = len(valid_molecules)
    validity_rate = num_valid / num_molecules

    print(f"Valid molecules: {num_valid}/{num_molecules} ({validity_rate*100:.1f}%)")
    print(f"  Parse errors: {parse_errors}")
    print(f"  Execution errors: {execution_errors}")

    # Track accuracy for each property type
    property_stats = {}
    for prop in property_token_ids.keys():
        property_stats[prop] = {'correct': 0, 'total': 0}

    overall_correct = 0
    overall_total = 0

    print(f"Calculating characterization token accuracy...")
    for idx, mol in enumerate(tqdm(valid_molecules, desc="Checking properties", unit="mol")):
        original_idx = valid_indices[idx]
        smiles = valid_smiles[idx]
        conditioned_props = molecule_properties[original_idx]

        if conditioned_props:
            try:
                props = calculate_properties(smiles)
                actual_props = props['quantized_properties']

                for prop in conditioned_props:
                    property_stats[prop]['total'] += 1
                    overall_total += 1
                    if prop in actual_props:
                        property_stats[prop]['correct'] += 1
                        overall_correct += 1

            except Exception as e:
                for prop in conditioned_props:
                    property_stats[prop]['total'] += 1
                    overall_total += 1

    print(f"\nCharacterization Token Accuracy:")
    print(f"  Overall: {overall_correct}/{overall_total} ({overall_correct/overall_total*100:.1f}%)" if overall_total > 0 else "  Overall: N/A")

    for prop, stats in sorted(property_stats.items()):
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total'] * 100
            print(f"  {prop}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")

    # Create 2D molecule images and log to wandb
    print(f"Creating molecule visualizations...")
    if num_valid > 0:
        mols_to_visualize = valid_molecules[:50]
        smiles_to_log = valid_smiles[:50]
        paths_to_log = valid_paths[:50]

        images = []
        print(f"Drawing {len(mols_to_visualize)} molecules...")
        for mol, smi in tqdm(zip(mols_to_visualize, smiles_to_log), desc="Drawing", total=len(mols_to_visualize), unit="mol"):
            try:
                img = Draw.MolToImage(mol, size=(300, 300))
                img_array = np.array(img)
                images.append(wandb.Image(img_array, caption=smi[:50]))
            except Exception as e:
                print(f"Error drawing molecule: {e}")
                continue

        print(f"Logging {len(images)} images to wandb...")
        if images:
            log_dict = {
                "generated_molecules": images,
                "validity_rate": validity_rate,
                "parse_error_rate": parse_errors / num_molecules,
                "execution_error_rate": execution_errors / num_molecules,
                "overall_conditioning_accuracy": overall_correct / overall_total if overall_total > 0 else 0.0,
            }

            for prop, stats in property_stats.items():
                if stats['total'] > 0:
                    prop_name = prop.replace('[', '').replace(']', '')
                    log_dict[f"{prop_name}_accuracy"] = stats['correct'] / stats['total']

            wandb.log(log_dict, step=step)
            print(f"Successfully logged images to wandb")

        # Log example synthesis paths as step-by-step images
        from synthesis.animate import parse_annotated_path
        from PIL import Image
        from io import BytesIO

        synthesis_images = []
        for path_str, product_smi in zip(paths_to_log[:5], smiles_to_log[:5]):
            try:
                # Re-execute to get annotated path with products
                _, annotated = execute_path(path_str, reactions)
                if not annotated:
                    continue

                steps = parse_annotated_path(annotated)
                if not steps:
                    continue

                # Create image for each step
                step_images = []
                for i, synth_step in enumerate(steps):
                    reactant_mols = [Chem.MolFromSmiles(r) for r in synth_step['reactants']]
                    reactant_mols = [m for m in reactant_mols if m]
                    product_mol = Chem.MolFromSmiles(synth_step['product'])
                    rxn_name = synth_step.get('reaction', '?')

                    if not reactant_mols or not product_mol:
                        continue

                    # Draw reactants + product with reaction name
                    # Legend: R1, R2, ... --[rxn_name]--> Product
                    all_mols = reactant_mols + [product_mol]
                    reactant_labels = [f"R{j+1}" for j in range(len(reactant_mols))]
                    legends = reactant_labels + [f"--[{rxn_name}]-->"]
                    img = Draw.MolsToGridImage(all_mols, molsPerRow=len(all_mols),
                                               subImgSize=(200, 200), legends=legends)
                    step_images.append(img)

                if step_images:
                    # Combine step images vertically
                    total_height = sum(img.height for img in step_images)
                    max_width = max(img.width for img in step_images)
                    combined = Image.new('RGB', (max_width, total_height), 'white')
                    y_offset = 0
                    for img in step_images:
                        combined.paste(img, (0, y_offset))
                        y_offset += img.height
                    synthesis_images.append(wandb.Image(combined, caption=f"Product: {product_smi[:50]}"))
            except Exception as e:
                print(f"Error creating synthesis animation: {e}")
                continue

        if synthesis_images:
            wandb.log({"synthesis_animations": synthesis_images}, step=step)
            print(f"Logged {len(synthesis_images)} synthesis animations to wandb")

    print(f"Setting model back to train mode...")
    model.train()

    return {
        "num_valid": num_valid,
        "num_invalid": invalid_count,
        "validity_rate": validity_rate,
        "valid_smiles": valid_smiles,
        "valid_paths": valid_paths,
        "parse_errors": parse_errors,
        "execution_errors": execution_errors,
        "overall_conditioning_accuracy": overall_correct / overall_total if overall_total > 0 else 0.0,
        "overall_conditioning_correct": overall_correct,
        "overall_conditioning_total": overall_total,
        "property_stats": property_stats,
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

    parser.add_argument("--batch_size", type=int, default=220)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--checkpoint", type=str, default='')
    parser.add_argument("--compile", action="store_true", help="Compile model with torch.compile")

    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=1_000_000)
    parser.add_argument("--max_grad_norm", type=float, default=1.5)

    parser.add_argument("--max_len", type=int, default=150, help="Input len")

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
            project="robosean",
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

    # Create dataset and dataloader
    train_dataset = SynthesisPathDataset(
        file_list=mol_files,
        tokenizer=tokenizer,
        max_len=args.max_len,
        add_characterization_tokens=True,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        #prefetch_factor=2 if args.num_workers > 0 else None,
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

        # Generate and log molecules
        if (global_step % args.generate_every_k_steps == 0 and global_step > 0) or global_step == 10:
            # Synchronize all ranks before evaluation
            if world_size > 1:
                torch.distributed.barrier()

            if rank == 0:
                print(f"\n{'='*60}")
                print(f"Generating molecules at step {global_step}...")
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

                    generation_stats = generate_and_log_molecules(
                        model=eval_model,
                        tokenizer=tokenizer,
                        reactions=reactions,
                        num_molecules=args.num_generate,
                        temperature=args.generation_temperature,
                        top_p=args.generation_top_p,
                        max_tokens=args.max_len,
                        device=device,
                        step=global_step
                    )

                    print(f"Generation complete: {generation_stats['num_valid']}/{args.num_generate} valid molecules")
                    print(f"Validity rate: {generation_stats['validity_rate']*100:.1f}%")
                    if generation_stats['overall_conditioning_total'] > 0:
                        print(f"Overall conditioning accuracy: {generation_stats['overall_conditioning_accuracy']*100:.1f}%")

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

            save_path = f"s3://shvaibackups/robosean/{args.experiment}/{global_step}.pt"

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