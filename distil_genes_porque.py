import os
import argparse
import torch
import torch.distributed as dist
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
import time
from pathlib import Path
from utils.models import BidirectionalLlama, LlamaConfig
from tokenizers import Tokenizer

from s3torchconnector import S3Checkpoint

from datetime import timedelta
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class GeneDistillClassifier(nn.Module):
    """Encoder + classifier head for KL distillation."""
    def __init__(self, encoder_config, num_classes):
        super().__init__()
        self.encoder = BidirectionalLlama(encoder_config)
        self.classifier = nn.Linear(encoder_config.hidden_size, num_classes, bias=True)

        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        print(f"GeneDistillClassifier: {encoder_params + classifier_params:,} params (encoder: {encoder_params:,}, classifier: {classifier_params:,})")

    def forward(self, input_ids, attention_mask=None):
        encoder_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_out['hidden_state']
        cls_embedding = hidden_states[:, 0, :]  # CLS token at position 0
        logits = self.classifier(cls_embedding)
        return logits


class GeneDistillDataset(IterableDataset):
    """Dataset for KL distillation from parquet files with smiles and pred_logits."""
    def __init__(self, file_list, tokenizer, max_len):
        super().__init__()
        self.file_list = file_list
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_token_id = tokenizer.token_to_id("[PAD]")

    def _pad_and_mask(self, ids):
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        attention_mask = [1] * len(ids) + [0] * (self.max_len - len(ids))
        ids = ids + [self.pad_token_id] * (self.max_len - len(ids))
        return ids, attention_mask

    def _read_parquet(self, path):
        if path.startswith('s3://'):
            import s3fs
            fs = s3fs.S3FileSystem()
            with fs.open(path, 'rb') as f:
                return pd.read_parquet(f, columns=['smiles', 'pred_logits'])
        else:
            return pd.read_parquet(path, columns=['smiles', 'pred_logits'])

    def __iter__(self):
        while True:
            random.seed(time.time_ns())
            np.random.seed(int(time.time_ns() % 2**32))

            file_path = random.choice(self.file_list)
            df = self._read_parquet(file_path)
            df = df.sample(frac=1)
            print(f"Loaded {file_path} with {len(df)} samples")

            for _, row in df.iterrows():
                smiles = row['smiles']
                teacher_logits = row['pred_logits']

                if '.' in smiles:
                    continue

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

                if len(canonical_smiles) > self.max_len - 2:
                    continue

                token_ids = self.tokenizer.encode(f'[BOS]{canonical_smiles}[EOS]').ids
                token_ids, attention_mask = self._pad_and_mask(token_ids)

                if isinstance(teacher_logits, np.ndarray):
                    teacher_logits = teacher_logits.astype(np.float32)
                else:
                    teacher_logits = np.array(teacher_logits, dtype=np.float32)

                yield {
                    'input_ids': torch.tensor(token_ids, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.bool),
                    'teacher_logits': torch.tensor(teacher_logits, dtype=torch.float32),
                }


def load_test_data(test_file, tokenizer, max_len, max_samples=2000):
    """Load test set into memory for evaluation."""
    print(f"Loading test data from {test_file}...")

    if test_file.startswith('s3://'):
        import s3fs
        fs = s3fs.S3FileSystem()
        with fs.open(test_file, 'rb') as f:
            df = pd.read_parquet(f, columns=['smiles', 'pred_logits'])
    else:
        df = pd.read_parquet(test_file, columns=['smiles', 'pred_logits'])

    pad_token_id = tokenizer.token_to_id("[PAD]")

    def pad_and_mask(ids):
        if len(ids) > max_len:
            ids = ids[:max_len]
        attention_mask = [1] * len(ids) + [0] * (max_len - len(ids))
        ids = ids + [pad_token_id] * (max_len - len(ids))
        return ids, attention_mask

    samples = []
    for _, row in df.iterrows():
        smiles = row['smiles']
        teacher_logits = row['pred_logits']

        if '.' in smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        if len(canonical_smiles) > max_len - 2:
            continue

        token_ids = tokenizer.encode(f'[BOS]{canonical_smiles}[EOS]').ids
        token_ids, attention_mask = pad_and_mask(token_ids)

        if isinstance(teacher_logits, np.ndarray):
            teacher_logits = teacher_logits.astype(np.float32)
        else:
            teacher_logits = np.array(teacher_logits, dtype=np.float32)

        samples.append({
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.bool),
            'teacher_logits': torch.tensor(teacher_logits, dtype=torch.float32),
        })

        if len(samples) >= max_samples:
            break

    print(f"Loaded {len(samples)} test samples")
    return samples



def do_eval(test_samples, model, device, temperature=1.0):
    """Evaluate KL divergence on test set."""
    model.eval()
    total_kl = 0.0
    num_samples = 0

    batch_size = 64
    for i in range(0, len(test_samples), batch_size):
        batch = test_samples[i:i+batch_size]

        input_ids = torch.stack([s['input_ids'] for s in batch]).to(device)
        attention_mask = torch.stack([s['attention_mask'] for s in batch]).to(device)
        teacher_logits = torch.stack([s['teacher_logits'] for s in batch]).to(device)

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                student_logits = model(input_ids=input_ids, attention_mask=attention_mask)

                # KL divergence: KL(teacher || student)
                teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
                student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
                kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

                total_kl += kl.item() * len(batch)
                num_samples += len(batch)

    model.train()
    avg_kl = total_kl / num_samples
    print(f"    Eval KL: {avg_kl:.4f}")
    return {'eval_kl': avg_kl}


def setup_ddp():
    dist.init_process_group(backend='nccl', timeout=timedelta(minutes=60))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def get_file_list(data_dir, test_file_name='compound_perturbation_000.parquet'):
    """Get list of parquet files from S3 or local dir, excluding test file."""
    if data_dir.startswith('s3://'):
        import s3fs
        fs = s3fs.S3FileSystem()
        # Remove s3:// prefix for s3fs
        bucket_path = data_dir.replace('s3://', '')
        files = fs.ls(bucket_path)
        files = [f's3://{f}' for f in files if f.endswith('.parquet')]
    else:
        files = list(Path(data_dir).glob('*.parquet'))
        files = [str(f) for f in files]

    # Exclude test file
    train_files = [f for f in files if test_file_name not in f]
    test_file = [f for f in files if test_file_name in f]

    print(f"Found {len(train_files)} train files, {len(test_file)} test file(s)")
    return train_files, test_file[0] if test_file else None


def parse_args():
    parser = argparse.ArgumentParser(description="KL Distillation training for gene expression prediction")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--encoder_checkpoint", type=str, default='s3://shvaibackups/mol_porque/e768_l20_h12_decode1_constrast1_pubchemcont/160001.pt')
    parser.add_argument("--checkpoint", type=str, default='', help="Resume from this checkpoint")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=100_000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=20)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--intermediate_size_multiplier", type=float, default=4.0)
    parser.add_argument("--max_mol_size", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=20574)
    parser.add_argument("--temperature", type=float, default=1.0, help="KL distillation temperature")

    parser.add_argument("--data_dir", type=str, default='s3://shvaibackups/cellpainto_v3/finetune_models/well_level_split_t25_other/predictions/')
    parser.add_argument("--tokenizer_path", type=str, default="tokenizers/smiles_tokenizer_simple/tokenizer.json")
    parser.add_argument("--experiment", type=str, default="gene_distill_v1")

    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=1_000_000)
    parser.add_argument("--eval_samples", type=int, default=2000)

    return parser.parse_args()



def main():
    args = parse_args()

    # Setup distributed
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1:
        local_rank = setup_ddp()
        rank = dist.get_rank()
    else:
        local_rank = 0
        rank = 0

    device = torch.device(f'cuda:{local_rank}')

    if rank == 0:
        wandb.init(project="gene_distill", name=args.experiment, config=vars(args))

    # Load tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    # Get file lists
    train_files, test_file = get_file_list(args.data_dir)

    # Create encoder config
    encoder_config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=args.emb_dim,
        intermediate_size=int(args.emb_dim * args.intermediate_size_multiplier),
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        pad_token_id=tokenizer.token_to_id('[PAD]'),
        bos_token_id=tokenizer.token_to_id('[BOS]'),
        eos_token_id=tokenizer.token_to_id('[EOS]'),
        cls_token_id=tokenizer.token_to_id('[BOS]'),
        max_position_embeddings=args.max_mol_size,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        attention_dropout=0.0,
        rope_base=10_000,
        use_rope=True,
    )

    # Create model
    model = GeneDistillClassifier(encoder_config, args.num_classes)
    model = model.to(device)

    # Load pretrained encoder weights
    if args.encoder_checkpoint != '':
        print(f"Loading encoder from: {args.encoder_checkpoint}")
        if args.encoder_checkpoint.startswith('s3://'):
            with S3Checkpoint("us-west-2").reader(args.encoder_checkpoint) as reader:
                checkpoint = torch.load(reader, map_location='cpu')
        else:
            checkpoint = torch.load(args.encoder_checkpoint, map_location='cpu')

        # Handle compiled model prefix
        if any(k.startswith('_orig_mod.') for k in checkpoint.keys()):
            print("Removing _orig_mod. prefix...")
            checkpoint = {k.replace('_orig_mod.', '', 1): v for k, v in checkpoint.items()}

        # Extract only encoder weights
        encoder_state = {k.replace('encoder.', '', 1): v for k, v in checkpoint.items() if k.startswith('encoder.')}
        model.encoder.load_state_dict(encoder_state)
        del checkpoint
        torch.cuda.empty_cache()
        print("Encoder loaded successfully")

    # Load full checkpoint if resuming
    if args.checkpoint != '':
        print(f"Resuming from: {args.checkpoint}")
        if args.checkpoint.startswith('s3://'):
            with S3Checkpoint("us-west-2").reader(args.checkpoint) as reader:
                checkpoint = torch.load(reader, map_location='cpu')
        else:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')

        if any(k.startswith('_orig_mod.') for k in checkpoint.keys()):
            checkpoint = {k.replace('_orig_mod.', '', 1): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)
        del checkpoint
        torch.cuda.empty_cache()

    if args.compile:
        model = torch.compile(model)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Dataset and dataloader
    dataset = GeneDistillDataset(train_files, tokenizer, args.max_mol_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    # Load test data
    test_samples = None
    if rank == 0 and test_file:
        test_samples = load_test_data(test_file, tokenizer, args.max_mol_size, max_samples=args.eval_samples)

    # Training loop
    model.train()
    global_step = 0
    total_start_time = time.time()
    data_iter = iter(dataloader)

    if rank == 0:
        print(f"Starting training for {args.max_steps} steps...")


    while global_step < args.max_steps:
        time_start = time.time()
        optimizer.zero_grad()

        total_loss = 0.0

        for _ in range(args.grad_accum_steps):
            batch = next(data_iter)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            teacher_logits = batch['teacher_logits'].to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                student_logits = model(input_ids=input_ids, attention_mask=attention_mask)

                # KL divergence loss
                teacher_probs = F.softmax(teacher_logits / args.temperature, dim=-1)
                student_log_probs = F.log_softmax(student_logits / args.temperature, dim=-1)
                loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
                loss = loss / args.grad_accum_steps

            total_loss += loss.detach()
            loss.backward()

        # Sync gradients
        if world_size > 1:
            dist.barrier()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
        optimizer.step()

        time_taken = time.time() - time_start

        # Logging
        if rank == 0:
            samples_per_sec = args.batch_size * args.grad_accum_steps / time_taken
            total_time = time.time() - total_start_time
            total_samples_k = (global_step + 1) * args.batch_size * args.grad_accum_steps / 1000.0

            print(f"Step {global_step} | Loss: {total_loss.item():.4f} | Grad: {grad_norm:.3f} | "
                  f"{samples_per_sec:.1f} samples/s | Data: {total_samples_k:.2f}K | Time: {total_time/60:.1f}min")

            wandb.log({
                'train/loss': total_loss.item(),
                'train/grad_norm': grad_norm,
                'train/samples_per_sec': samples_per_sec,
                'train/lr': args.lr,
                'train/samples_k': total_samples_k,
            }, step=global_step)

        # Evaluation - use underlying module to avoid DDP sync issues
        if global_step > 0 and global_step % args.eval_every == 0:
            if world_size > 1:
                dist.barrier()
            if rank == 0 and test_samples:
                print(f"Running evaluation at step {global_step}...")
                eval_model = model.module if world_size > 1 else model
                eval_results = do_eval(test_samples, eval_model, device, args.temperature)
                wandb.log({'eval/kl': eval_results['eval_kl']}, step=global_step)
            if world_size > 1:
                dist.barrier()

        # Save checkpoint
        if rank == 0 and global_step > 0 and global_step % args.save_every == 0:
            if world_size > 1:
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()

            # Remove _orig_mod prefix if compiled
            if any(k.startswith('_orig_mod.') for k in model_state_dict.keys()):
                model_state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in model_state_dict.items()}

            save_path = f"s3://shvaibackups/mol_porque/{args.experiment}/{global_step}.pt"
            with S3Checkpoint("us-west-2").writer(save_path) as writer:
                torch.save(model_state_dict, writer)
            print(f"Saved checkpoint to {save_path}")

        if world_size > 1:
            dist.barrier()

        global_step += 1

    cleanup_ddp() if world_size > 1 else None


if __name__ == "__main__":
    main()
