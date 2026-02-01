import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import tqdm

class LlamaConfig:
    def __init__(
        self,
        pad_token_id,
        bos_token_id,
        eos_token_id,
        cls_token_id,
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        attention_dropout=0.0,
        rope_base=10_000,
        max_position_embeddings=2048,
        use_rope=True,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.attention_dropout = attention_dropout
        self.cls_token_id = cls_token_id
        self.rope_base = rope_base
        self.max_position_embeddings = max_position_embeddings
        self.use_rope = use_rope

class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, q, k, position_ids):
        # q, k: [bs, n_head, seq_len, head_dim]
        # position_ids: [bs, seq_len]
        
        # Create sinusoidal pattern
        t = position_ids.float().unsqueeze(-1) @ self.inv_freq.unsqueeze(0)
        freqs = torch.cat([t, t], dim=-1)
        
        # Apply rotary embeddings to q and k
        q_embed = self._apply_rotary_pos_emb(q, freqs)
        k_embed = self._apply_rotary_pos_emb(k, freqs)
        
        return q_embed, k_embed
    
    def _apply_rotary_pos_emb(self, x, freqs):
        # x: [bs, n_head, seq_len, head_dim]
        # freqs: [bs, seq_len, head_dim]
        
        # Reshape for broadcasting
        freqs = freqs.unsqueeze(1)  # [bs, 1, seq_len, head_dim]
        
        # Split x into real and imaginary parts (even and odd dimensions)
        x_real, x_imag = x.chunk(2, dim=-1)
        
        # Apply complex multiplication
        # Make sure freqs has the right dimension
        freqs_cos = torch.cos(freqs)[..., :x_real.shape[-1]]
        freqs_sin = torch.sin(freqs)[..., :x_real.shape[-1]]
        
        x_out_real = x_real * freqs_cos - x_imag * freqs_sin
        x_out_imag = x_real * freqs_sin + x_imag * freqs_cos
        
        # Concatenate real and imaginary parts
        x_out = torch.cat([x_out_real, x_out_imag], dim=-1)
        
        return x_out

class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # RoPE parameters
        self.rotary_emb = None
        if config.use_rope:
            self.rotary_emb = RotaryEmbedding(config)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, output_attentions=False):
        bsz, seq_len, _ = hidden_states.size()
        
        # Default position_ids if not provided
        if self.rotary_emb:
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        if self.rotary_emb:
            q, k = self.rotary_emb(q, k, position_ids)
        
        if attention_mask is None:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
            )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if output_attentions:
            return attn_output, None  # Replace None with actual attention weights if needed
        return attn_output

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(self.weight.dtype)

class LlamaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, output_attentions=False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        if output_attentions:
            hidden_states, attn_weights = self.self_attn(hidden_states, attention_mask=attention_mask, position_ids=position_ids, output_attentions=True)
        else:
            hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
        
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if output_attentions:
            return hidden_states, attn_weights
        return hidden_states

class BidirectionalLlama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        #self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layers = nn.ModuleList([LlamaLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, 
        input_ids, 
        labels=None, 
        attention_mask=None, 
    ):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        
        attention_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)

        hidden_states = self.embed_tokens(input_ids)
        
        # Create position IDs
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)
        
        hidden_states = hidden_states #+ self.embed_positions(position_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        output_dict = {"logits": logits}
        output_dict['hidden_state'] = hidden_states

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
            output_dict["mlm_loss"] = loss

        return output_dict        


class EmbeddingConditionalLlamaDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        #self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size) # In this house we use rope
        self.layers = nn.ModuleList([LlamaLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.stop_token_id = config.eos_token_id

    def forward(self, input_ids, attention_mask=None, position_ids=None, conditioning_embeddings=None, labels=None):
        batch_size, seq_length = input_ids.size()

        # Embed input tokens
        hidden_states = self.embed_tokens(input_ids)

        # Handle conditioning embeddings - prepend to sequence
        cond_seq_len = 0
        if conditioning_embeddings is not None:
            # conditioning_embeddings: [bs, hidden_size] -> [bs, 1, hidden_size]
            if conditioning_embeddings.dim() == 2:
                conditioning_embeddings = conditioning_embeddings.unsqueeze(1)
            cond_seq_len = conditioning_embeddings.size(1)
            # Prepend conditioning to hidden states
            hidden_states = torch.cat([conditioning_embeddings, hidden_states], dim=1)

        total_seq_len = cond_seq_len + seq_length

        # Create causal attention mask for full sequence (conditioning + input)
        if attention_mask is not None:
            # attention_mask provided is for input_ids only, need to extend for conditioning
            # Assume attention_mask is [bs, seq_length] padding mask
            if attention_mask.dim() == 2:
                # Prepend 1s for conditioning tokens (always attend to them)
                cond_mask = torch.ones(batch_size, cond_seq_len, device=input_ids.device, dtype=attention_mask.dtype)
                extended_mask = torch.cat([cond_mask, attention_mask], dim=1)  # [bs, total_seq_len]
                # Create causal mask
                causal_mask = torch.tril(torch.ones(total_seq_len, total_seq_len, device=input_ids.device)).bool()
                # Combine: [bs, 1, 1, total_seq_len] * [1, 1, total_seq_len, total_seq_len]
                padding_mask = extended_mask.bool().unsqueeze(1).unsqueeze(2)
                attention_mask = causal_mask.unsqueeze(0) & padding_mask
                attention_mask = attention_mask.expand(batch_size, 1, total_seq_len, total_seq_len)
            elif attention_mask.dim() == 4:
                # Already [bs, 1, seq, seq], extend for conditioning
                # Conditioning tokens can attend to themselves and each other
                # Input tokens can attend to conditioning + causal input
                cond_self = torch.ones(batch_size, 1, cond_seq_len, cond_seq_len, device=input_ids.device, dtype=attention_mask.dtype)
                cond_to_input = torch.zeros(batch_size, 1, cond_seq_len, seq_length, device=input_ids.device, dtype=attention_mask.dtype)
                input_to_cond = torch.ones(batch_size, 1, seq_length, cond_seq_len, device=input_ids.device, dtype=attention_mask.dtype)
                # Build full mask
                top_row = torch.cat([cond_self, cond_to_input], dim=-1)  # [bs, 1, cond_seq_len, total_seq_len]
                bottom_row = torch.cat([input_to_cond, attention_mask], dim=-1)  # [bs, 1, seq_length, total_seq_len]
                attention_mask = torch.cat([top_row, bottom_row], dim=2)  # [bs, 1, total_seq_len, total_seq_len]

        # Create position IDs - conditioning tokens get positions 0..cond_seq_len-1
        # Input tokens get positions cond_seq_len..total_seq_len-1
        if position_ids is None:
            position_ids = torch.arange(total_seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        elif cond_seq_len > 0:
            # Extend provided position_ids for conditioning
            cond_positions = torch.arange(cond_seq_len, dtype=torch.long, device=input_ids.device)
            cond_positions = cond_positions.unsqueeze(0).expand(batch_size, -1)
            # Shift input position_ids by cond_seq_len
            position_ids = torch.cat([cond_positions, position_ids + cond_seq_len], dim=1)
        
        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids)

        hidden_states = self.norm(hidden_states)

        # Compute logits for full sequence (conditioning + input)
        # Conditioning token predicts first input token
        logits = self.lm_head(hidden_states)

        output_dict = {
            "logits": logits,
            "last_hidden_states": hidden_states,
            "cond_seq_len": cond_seq_len,
        }

        # Compute loss if labels are provided
        if labels is not None:
            # Prepend -100 for conditioning tokens so they align properly
            # cond_logit[0] predicts labels[0], cond_logit[1] predicts labels[1], etc.
            if cond_seq_len > 0:
                ignore_prefix = torch.full((batch_size, cond_seq_len), -100, dtype=torch.long, device=labels.device)
                labels = torch.cat([ignore_prefix, labels], dim=1)

            labels = labels.masked_fill(labels == self.config.pad_token_id, -100)

            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Compute cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            batch_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            loss = batch_loss.sum() / (shift_labels != -100).sum()
            output_dict["loss"] = loss
            output_dict["batch_loss"] = torch.stack([losses.sum() / (lbls != -100).sum() for losses, lbls in zip(batch_loss.reshape(batch_size, -1), shift_labels)])

        return output_dict

    def generate(self,
        input_ids_list,
        conditioning_embeddings=None,
        max_generated_tokens=None,
        stop_token_id=None,
        temperature=0.0,
        top_p=0.9,
        logit_biases=None,
    ):
        if max_generated_tokens is None:
            max_generated_tokens = self.config.max_position_embeddings - 3

        if stop_token_id is None:
            stop_token_id = self.stop_token_id

        device = next(self.parameters()).device
        batch_size = len(input_ids_list)

        # Convert input_ids_list to list of lists for easier manipulation
        generated_tokens = [tensor.tolist() for tensor in input_ids_list]

        # Track which sequences have finished generating
        finished_sequences = [False] * batch_size

        for _ in tqdm.tqdm(range(max_generated_tokens), desc="Generating", total=max_generated_tokens):
            # Skip if all sequences are finished
            if all(finished_sequences):
                break

            # Find the maximum length in the current batch
            max_len = max(len(seq) for seq in generated_tokens)

            # Left pad all sequences to the same length
            padded_sequences = []
            attention_masks = []

            for seq in generated_tokens:
                pad_length = max_len - len(seq)
                # Left pad with pad_token_id (assuming it exists in config)
                pad_token_id = getattr(self.config, 'pad_token_id', 0)
                padded_seq = [pad_token_id] * pad_length + seq
                padded_sequences.append(padded_seq)

                # Create attention mask: 0 for padding, 1 for real tokens
                attention_mask = [0] * pad_length + [1] * len(seq)
                attention_masks.append(attention_mask)

            # Convert to tensors
            current_input_ids = torch.tensor(padded_sequences, dtype=torch.long, device=device)
            attention_mask = torch.tensor(attention_masks, dtype=torch.bool, device=device)

            # Create position_ids - for left padding, position_ids should start from 0 for the first real token
            position_ids = torch.zeros_like(current_input_ids, dtype=torch.long, device=device)
            for i, mask in enumerate(attention_masks):
                # Find where real tokens start (first 1 in attention mask)
                real_token_start = mask.index(1) if 1 in mask else 0
                # Set position_ids starting from 0 for real tokens
                for j in range(real_token_start, len(mask)):
                    position_ids[i, j] = j - real_token_start

            with torch.no_grad():
                outputs = self.forward(
                    input_ids=current_input_ids,
                    conditioning_embeddings=conditioning_embeddings.to(device) if conditioning_embeddings is not None else None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    labels=None,
                )

            logits = outputs['logits']
            if logit_biases is not None:
                logits = logits + logit_biases.unsqueeze(0).unsqueeze(0).expand(logits.size(0), logits.size(1), -1)
            next_token_logits = logits[:, -1, :]  # Get last token logits for each sequence

            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Apply the mask to each sequence in the batch
                for idx in range(batch_size):
                    indices_to_remove = sorted_indices[idx][sorted_indices_to_remove[idx]]
                    next_token_logits[idx, indices_to_remove] = float('-inf')

            # Sample from the filtered distribution
            if temperature == 0:  # greedy
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            else:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Handle finished sequences
            for idx in range(batch_size):
                if finished_sequences[idx]:
                    next_tokens[idx] = stop_token_id
                elif next_tokens[idx] == stop_token_id:
                    finished_sequences[idx] = True

                generated_tokens[idx].append(next_tokens[idx].item())

            # Check if we've hit max length
            if max(len(seq) for seq in generated_tokens) >= (self.config.max_position_embeddings-1):
                break

        # Return the generated sequences
        return generated_tokens

def create_custom_llama_config(**kwargs):
    return LlamaConfig(**kwargs)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def ema_update(ema_sd, model, ema_decay=.999):
    with torch.no_grad():
        for k, v in model.module.named_buffers():
            ema_sd[k] = v
        for k, v in model.module.named_parameters():
            ema_sd[k].data.mul_(ema_decay).add_(v.data, alpha=1 - ema_decay)
    
    return ema_sd

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time step embeddings for diffusion models."""
    def __init__(self, hidden_size, max_period=10000):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_period = max_period
        # MLP to project time embedding to hidden size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, time_steps):
        """
        Args:
            time_steps: [bs] or [bs, 1] - diffusion time steps (0 to 1 or 0 to T)
        Returns:
            time_emb: [bs, hidden_size]
        """
        if time_steps.dim() == 2:
            time_steps = time_steps.squeeze(-1)  # [bs]

        half_dim = self.hidden_size // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=time_steps.device, dtype=torch.float32) / half_dim
        )
        # [bs, half_dim]
        args = time_steps[:, None].float() * freqs[None, :]
        # [bs, hidden_size]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # Project through MLP
        emb = self.mlp(emb)
        return emb


class LlamaDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.time_emb = SinusoidalTimeEmbedding(config.hidden_size)
        self.layers = nn.ModuleList([LlamaLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self,
        noised_embs,
        time_steps,
        attention_mask=None,
    ):
        """
        Args:
            noised_embs: [bs, seq_len, hidden_size] - noised input embeddings
            time_steps: [bs] or [bs, 1] - diffusion time steps
            attention_mask: [bs, seq_len] - attention mask (1 = attend, 0 = ignore)
        Returns:
            dict with 'hidden_state': [bs, seq_len, hidden_size]
        """
        batch_size, seq_length, _ = noised_embs.size()
        device = noised_embs.device

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=device)

        attention_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)

        # Create position IDs
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Sinusoidal time embedding: [bs, hidden_size] -> [bs, 1, hidden_size]
        time_emb = self.time_emb(time_steps).unsqueeze(1)

        # Add time embedding to all positions
        hidden_states = noised_embs + time_emb

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids)

        hidden_states = self.norm(hidden_states)

        output_dict = {'hidden_state': hidden_states}
        return output_dict


class FragMol(nn.Module):
    def __init__(self, encoder, decoder, diffusion_model):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.diffusion_model = diffusion_model

    def forward(self, fragment_ids, fragment_attention_mask, fragment_to_mol_map, max_mol_size=5):
        """
        Args:
            fragment_ids: [num_fragments, seq_len] - token ids for each fragment
            fragment_attention_mask: [num_fragments, seq_len] - attention mask for fragments
            fragment_to_mol_map: [num_fragments] - which molecule each fragment belongs to
            max_mol_size: int - max number of fragments per molecule
        """
        device = fragment_ids.device
        hidden_dim = self.encoder.config.hidden_size

        # Encode fragments -> get CLS embeddings
        encoder_out = self.encoder(
            input_ids=fragment_ids,
            attention_mask=fragment_attention_mask
        )
        fragment_cls = encoder_out['hidden_state'][:, 0, :]  # [num_fragments, hidden_dim]

        # Build molecule-level tensors
        num_mols = len(np.unique(fragment_to_mol_map))
        true_hiddens = torch.zeros(num_mols, max_mol_size, hidden_dim, device=device)
        diffusion_attn_mask = torch.zeros(num_mols, max_mol_size, device=device)

        for mol_idx in np.unique(fragment_to_mol_map):
            # Get fragment indices for this molecule
            frag_indices = np.where(fragment_to_mol_map == mol_idx)[0]
            num_frags = min(len(frag_indices), max_mol_size)

            # Fill true_hiddens with fragment embeddings
            true_hiddens[mol_idx, :num_frags] = fragment_cls[frag_indices[:num_frags]]
            # Set attention mask (1 for real fragments, 0 for padding)
            diffusion_attn_mask[mol_idx, :num_frags] = 1.0

        # Flow matching: noise -> true_hiddens
        noise = torch.randn_like(true_hiddens)
        flow_true = true_hiddens - noise  # target velocity: from noise to data

        # Random time steps between 0 and 1, with 2% chance of t=0
        time_steps = torch.rand(num_mols, device=device)
        zero_mask = torch.rand(num_mols, device=device) < 0.02
        time_steps = torch.where(zero_mask, torch.zeros_like(time_steps), time_steps)

        # Interpolate: observed = t * true_hiddens + (1-t) * noise
        t = time_steps[:, None, None]  # [num_mols, 1, 1]
        observed = t * true_hiddens + (1 - t) * noise

        # Predict flow
        flow_pred_out = self.diffusion_model(
            noised_embs=observed,
            time_steps=time_steps,
            attention_mask=diffusion_attn_mask
        )
        flow_pred = flow_pred_out['hidden_state']

        # Flow loss: MSE weighted by attention mask
        flow_diff = (flow_pred - flow_true) ** 2  # [num_mols, max_mol_size, hidden_dim]
        flow_loss = (flow_diff * diffusion_attn_mask).sum() / diffusion_attn_mask.sum()

        # Decoder: reconstruct fragments conditioned on their CLS embeddings
        decoder_out = self.decoder(
            input_ids=fragment_ids,
            attention_mask=fragment_attention_mask,
            conditioning_embeddings=fragment_cls,
            labels=fragment_ids
        )

        output_dict = {
            'flow_loss': flow_loss,
            'decoder_loss': decoder_out['loss'],
        }

        return output_dict


class Smile2SmileVAE(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super().__init__()
        self.encoder = BidirectionalLlama(encoder_config)
        self.decoder = EmbeddingConditionalLlamaDecoder(decoder_config)
        self.mu_proj = nn.Linear(encoder_config.hidden_size, encoder_config.hidden_size, bias=False)
        self.logvar_proj = nn.Linear(encoder_config.hidden_size, encoder_config.hidden_size, bias=True)
        self.max_len_to_gen = decoder_config.max_position_embeddings
        self.eos_token_id = decoder_config.eos_token_id
        self.bos_token_id = decoder_config.bos_token_id
        self.cls_token_id = decoder_config.cls_token_id
        self.contrastive_temp = torch.nn.Parameter(torch.tensor(1.0))
        self.contrastive_bias = torch.nn.Parameter(torch.tensor(0.1))
        self.jepa_nn = TwoLayerNN(input_size=encoder_config.hidden_size, output_size=encoder_config.hidden_size)

        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = encoder_params + decoder_params
        import torch.distributed as dist
        print(f"Initialized! {total_params:,} total parameters (encoder: {encoder_params:,} decoder: {decoder_params:,})")


    def encode(self,
        input_ids,
        attention_mask=None,
        noise_ratio=1.0,
    ):
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        encoded_hidden_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )['hidden_state']

        # Use CLS token (position 0) for VAE
        cls_hidden = encoded_hidden_states[:, 0, :]
        mu = self.mu_proj(cls_hidden)
        logvars = self.logvar_proj(cls_hidden)

        # Reparameterization trick
        latent = noise_ratio * torch.randn_like(mu) * torch.exp(0.5 * logvars) + mu

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvars - mu.pow(2) - logvars.exp(), dim=1).mean()

        return {
            'latent': latent.unsqueeze(1),  # [bs, 1, hidden_size] for conditioning
            'kl_loss': kl_loss,
            'mu_norm': torch.norm(mu, dim=1).mean().item(),
            'std_latents': torch.exp(0.5 * logvars).mean().item(),
            'hidden_states': encoded_hidden_states,
        }

    def decode(self, latent, initial_tokens=None, temperature=0.0, top_p=0.9):
        device = next(self.parameters()).device
        bs, seqlen, dim = latent.shape

        if initial_tokens is None:
            initial_tokens = torch.LongTensor([self.cls_token_id, self.bos_token_id]).unsqueeze(0).expand(bs, 2)

        # generate() expects a list of token tensors
        input_ids_list = [initial_tokens[i].to(device) for i in range(bs)]
        generated_output = self.decoder.generate(
            input_ids_list=input_ids_list,
            conditioning_embeddings=latent,
            stop_token_id=self.eos_token_id,
            max_generated_tokens=self.max_len_to_gen,
            temperature=temperature,
            top_p=top_p,
        )

        return generated_output

    def forward(self,
            canonical_input_ids,
            canonical_attention_mask,
            random_view_input_ids,
            random_view_attention_mask,
            p_downscale_noise=0.05,
        ):
        bs = canonical_input_ids.size(0)

        # Occasionally reduce noise for curriculum learning
        noise_ratio = 1.0
        if np.random.rand() < p_downscale_noise:
            noise_ratio = np.random.rand()
            if np.random.rand() < 0.1:
                noise_ratio = 0.0

        noise_ratio_random = 1.0
        if np.random.rand() < p_downscale_noise:
            noise_ratio_random = np.random.rand()
            if np.random.rand() < 0.1:
                noise_ratio_random = 0.0

        # Encode both views
        canonical_encode = self.encode(
            input_ids=canonical_input_ids,
            attention_mask=canonical_attention_mask,
            noise_ratio=noise_ratio,
        )

        random_encode = self.encode(
            input_ids=random_view_input_ids,
            attention_mask=random_view_attention_mask,
            noise_ratio=noise_ratio_random,
        )

        canonical_latents = canonical_encode['latent']  # [bs, 1, hidden]
        random_latents = random_encode['latent']  # [bs, 1, hidden]

        # Decode both views into canonical SMILES
        canonical_decode = self.decoder(
            input_ids=canonical_input_ids,
            attention_mask=canonical_attention_mask,
            conditioning_embeddings=canonical_latents,
            labels=canonical_input_ids.clone()
        )

        random_decode = self.decoder(
            input_ids=canonical_input_ids,
            attention_mask=canonical_attention_mask,
            conditioning_embeddings=random_latents,
            labels=canonical_input_ids.clone()
        )

        decoder_loss = canonical_decode['loss'] + random_decode['loss']
        kl_loss = canonical_encode['kl_loss'] + random_encode['kl_loss']

        mu_norm = (canonical_encode['mu_norm'] + random_encode['mu_norm']) / 2
        std_latents = (canonical_encode['std_latents'] + random_encode['std_latents']) / 2

        # Contrastive loss (SigLIP formulation)
        predicted_canonical_latents = self.jepa_nn(random_latents.squeeze(1)) + random_latents.squeeze(1)
        canonical_latents_norm = F.normalize(canonical_latents.squeeze(1), dim=-1)  # [bs, hidden]
        random_latents_norm = F.normalize(predicted_canonical_latents, dim=-1)  # [bs, hidden]

        sims = canonical_latents_norm @ random_latents_norm.T  # [bs, bs]
        sims = sims * torch.exp(self.contrastive_temp) + self.contrastive_bias

        # Target: diagonal should be 1 (same molecule), off-diagonal 0
        targets = torch.eye(bs, device=sims.device)

        # Binary cross-entropy with logits
        contrastive_loss = F.binary_cross_entropy_with_logits(sims, targets)

        out = {
            'decoder_loss': decoder_loss,
            'contrastive_loss': contrastive_loss,
            'kl_loss': kl_loss,
            'std_latents': std_latents,
            'mu_norm': mu_norm,
            'loss': decoder_loss + kl_loss + contrastive_loss,
        }

        return out

class TwoLayerNN(nn.Module):
    def __init__(self, input_size, hidden_mult=2, output_size=1):
        super(TwoLayerNN, self).__init__()
        # For SwiGLU, we need to split the hidden layer into two parts: one for the gate and one for the value
        hidden_size = input_size * hidden_mult
        # The first layer projects to 2x hidden size (for gate and value)
        self.fc1 = nn.Linear(input_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Apply first linear layer
        x = self.fc1(x)
        # Split into gate and value
        hidden_size = x.shape[-1] // 2
        gate, value = x[..., :hidden_size], x[..., hidden_size:]
        # Apply SwiGLU: gate * swish(value)
        # swish(x) = x * sigmoid(x)
        x = gate * (value * torch.sigmoid(value))
        # Apply final linear layer
        x = self.fc2(x)
        return x



def jepa_vicreg_loss(target, pred):
    batch_size, dim = target.shape
    
    # ========== 1. JEPA LOSS (Invariance) ==========
    jepa_loss = F.mse_loss(pred, target.detach())
    
    # ========== 2. VARIANCE LOSS ==========
    # Compute std for predictions
    pred_std = torch.sqrt(pred.var(dim=0) + 1e-4)
    var_loss_pred = torch.relu(1.0 - pred_std).mean()
    
    # Compute std for targets
    target_std = torch.sqrt(target.var(dim=0) + 1e-4)
    var_loss_target = torch.relu(1.0 - target_std).mean()
    
    var_loss = (var_loss_pred + var_loss_target) / 2
    
    # ========== 3. COVARIANCE LOSS ==========
    def compute_cov_loss(z):
        """Decorrelate dimensions by penalizing off-diagonal covariance"""
        # Center the embeddings
        z_centered = z - z.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix
        cov = (z_centered.T @ z_centered) / (batch_size - 1)
        
        # Sum squared off-diagonal elements, normalized by dimension
        off_diag_mask = ~torch.eye(dim, device=z.device, dtype=torch.bool)
        return (cov[off_diag_mask] ** 2).sum() / dim
    
    cov_loss_pred = compute_cov_loss(pred)
    cov_loss_target = compute_cov_loss(target)
    cov_loss = (cov_loss_pred + cov_loss_target) / 2
    
    return {
        'jepa_loss': jepa_loss,
        'var_loss': var_loss,
        'cov_loss': cov_loss
    }


class Smile2SmileEncoderWithJEPA(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super().__init__()
        self.encoder = BidirectionalLlama(encoder_config)
        self.decoder = EmbeddingConditionalLlamaDecoder(decoder_config)
        self.max_len_to_gen = decoder_config.max_position_embeddings
        self.eos_token_id = decoder_config.eos_token_id
        self.bos_token_id = decoder_config.bos_token_id
        self.cls_token_id = decoder_config.cls_token_id
        self.contrastive_temp = torch.nn.Parameter(torch.tensor(1.0))
        self.contrastive_bias = torch.nn.Parameter(torch.tensor(0.1))
        self.jepa_nn = TwoLayerNN(input_size=encoder_config.hidden_size, output_size=encoder_config.hidden_size)

        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = encoder_params + decoder_params
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"{total_params:,} total parameters (encoder: {encoder_params:,} decoder: {decoder_params:,})")

    def decode(self, latent, initial_tokens=None, temperature=0.0, top_p=0.9):
        device = next(self.parameters()).device
        bs, seqlen, dim = latent.shape

        if initial_tokens is None:
            initial_tokens = torch.LongTensor([self.cls_token_id, self.bos_token_id]).unsqueeze(0).expand(bs, 2)

        # generate() expects a list of token tensors
        input_ids_list = [initial_tokens[i].to(device) for i in range(bs)]
        generated_output = self.decoder.generate(
            input_ids_list=input_ids_list,
            conditioning_embeddings=latent,
            stop_token_id=self.eos_token_id,
            max_generated_tokens=self.max_len_to_gen,
            temperature=temperature,
            top_p=top_p,
        )

        return generated_output

    def forward(self,
            canonical_input_ids,
            canonical_attention_mask,
            view1_input_ids,
            view1_attention_mask,
            view2_input_ids,
            view2_attention_mask,
        ):
        bs = canonical_input_ids.size(0)

        # Encode both views
        view1_encode = self.encoder(
            input_ids=view1_input_ids,
            attention_mask=view1_attention_mask,
        )

        view2_encode = self.encoder(
            input_ids=view2_input_ids,
            attention_mask=view2_attention_mask,
        )

        view1_cls = view1_encode['hidden_state'][:, 0, :].squeeze()
        view2_cls = view2_encode['hidden_state'][:, 0, :].squeeze()

        # Decode both views into canonical SMILES
        view1_decode = self.decoder(
            input_ids=canonical_input_ids,
            attention_mask=canonical_attention_mask,
            conditioning_embeddings=view1_cls,
            labels=canonical_input_ids.clone()
        )

        decoder_loss = view1_decode['loss'] 

        view1_cls_token_pred = self.jepa_nn(view2_cls) + view2_cls

        ## JEPA stuff
        jepa_losses = jepa_vicreg_loss(target=view1_cls.squeeze(), pred=view1_cls_token_pred.squeeze())
        jepa_loss = jepa_losses['jepa_loss'] + 5*jepa_losses['var_loss'] + 5*jepa_losses['cov_loss']

        # Contrastive loss (SigLIP formulation)
        view1_cls_norm = F.normalize(view1_cls, dim=-1)  # [bs, hidden]
        view1_cls_pred_norm = F.normalize(view1_cls_token_pred, dim=-1)  # [bs, hidden]

        sims = view1_cls_norm @ view1_cls_pred_norm.T  # [bs, bs]
        sims = sims * torch.exp(self.contrastive_temp) + self.contrastive_bias

        # Target: diagonal should be 1 (same molecule), off-diagonal 0
        targets = torch.eye(bs, device=sims.device)

        # Binary cross-entropy with logits
        contrastive_loss = F.binary_cross_entropy_with_logits(sims, targets)

        out = {
            'decoder_loss': decoder_loss,
            'contrastive_loss': contrastive_loss,
            'jepa_loss': jepa_loss,
        }

        return out

class PharmocophoreEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.coordinate_projection = nn.Linear(3, config.hidden_size)

    def forward(self,
        pharmacophore_coords,
        pharmacophore_type_ids,
        attention_mask=None,
    ):
        """
        Args:
            pharmacophore_coords: [bs, seq_len, 3] - 3D coordinates
            pharmacophore_type_ids: [bs, seq_len] - type IDs (1-indexed, 0=pad)
            attention_mask: [bs, seq_len] - 1 for real, 0 for pad
        Returns:
            hidden_states: [bs, seq_len, hidden_size]
        """
        batch_size, seq_length = pharmacophore_type_ids.size()

        if attention_mask is None:
            # Create mask from type_ids (0 = pad)
            attention_mask = (pharmacophore_type_ids != 0).long()

        # Expand for attention: [bs, 1, 1, seq_len]
        attention_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)

        hidden_states = self.embed_tokens(pharmacophore_type_ids) + self.coordinate_projection(pharmacophore_coords)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        hidden_states = self.norm(hidden_states)

        return hidden_states