"""Module implementation."""
import torch
from typing import Optional, Callable, Union, Iterable
from .kv_cache import KVCache
from .stop_criteria import StopCriteria


def autoregressive_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 20,
    eos_token_id: Optional[Union[int, Iterable[int]]] = None,
    sampler_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    use_cache: bool = False,
) -> torch.Tensor:
    """
    Generate tokens autoregressively.
    input_ids shape: (batch_size, seq_len)
    """
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    generated = input_ids.clone()
    batch_size = generated.size(0)
    stop_criteria = StopCriteria(max_length=generated.size(1) + max_new_tokens, eos_token_id=eos_token_id)

    if use_cache:
        config = getattr(model, "config", None)
        num_layers = getattr(model, "num_layers", getattr(config, "n_layers", 12))
        kv_cache = KVCache(num_layers=num_layers)
    else:
        kv_cache = None

    for step in range(max_new_tokens):
        if kv_cache is not None:
            model_inputs = generated if step == 0 else generated[:, -1:]
            logits = model(model_inputs, kv_cache=kv_cache)
        else:
            logits = model(generated)

        # Take logits for the last token position: (batch_size, vocab_size)
        if isinstance(logits, tuple):
            logits = logits[0]
        next_token_logits = logits[:, -1, :]

        if sampler_fn is not None:
            next_token = sampler_fn(next_token_logits)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        if next_token.dim() == 1:
            next_token = next_token.unsqueeze(-1)

        generated = torch.cat([generated, next_token], dim=-1)

        last_id = next_token[0, 0].item() if batch_size == 1 else None
        if stop_criteria.should_stop(generated.size(1), last_id):
            break
        eos_ids = (
            set(eos_token_id) if isinstance(eos_token_id, (list, tuple, set))
            else ({eos_token_id} if eos_token_id is not None else set())
        )
        if batch_size > 1 and eos_ids and torch.isin(
            next_token, torch.tensor(list(eos_ids), device=next_token.device)
        ).all():
            break

    return generated
