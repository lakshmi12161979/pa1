import torch
from model.gpt import GPT

# =========================
# SIMPLE GENERATION FUNCTION
# =========================
def generate(model, start_tokens, max_new_tokens, context_length):
    model.eval()

    tokens = start_tokens

    for _ in range(max_new_tokens):
        tokens_cond = tokens[-context_length:]

        logits = model(tokens_cond.unsqueeze(0))  # (1, T, C)
        logits = logits[:, -1, :]  # last token

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat([tokens, next_token.squeeze(0)], dim=0)

    return tokens


# =========================
# MAIN RUN
# =========================
if __name__ == "__main__":
    print("🚀 Generating text...")

    # Same config as training
    vocab_size = 100
    context_length = 10

    model = GPT(
        vocab_size=vocab_size,
        context_length=context_length,
        model_dim=64,
        num_blocks=2,
        num_heads=2
    )

    # ⚠️ No trained weights yet → random output
    start = torch.randint(0, vocab_size, (5,))

    output = generate(model, start, max_new_tokens=20, context_length=context_length)

    print("Generated tokens:")
    print(output.tolist())
