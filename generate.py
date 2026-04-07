import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(
        self,
        model,
        new_chars: int,
        context: TensorType[int],
        context_length: int,
        int_to_char: dict
    ) -> str:
        generator = torch.manual_seed(0)
        initial_state = generator.get_state()

        result = []
        for i in range(new_chars):
            # 1. Crop context
            if context.size(1) > context_length:
                context = context[:, -context_length:]

            # 2. Forward pass
            logits = model(context)  # (B, T, vocab_size)
            last_logits = logits[:, -1, :]  # (B, vocab_size)
            probs = torch.softmax(last_logits, dim=-1)

            # 3. Sample next token (reproducible)
            generator.set_state(initial_state)
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)

            # 4. Append to context
            context = torch.cat([context, next_token], dim=1)

            # 5. Map to character
            result.append(int_to_char[next_token.item()])

        return "".join(result)
