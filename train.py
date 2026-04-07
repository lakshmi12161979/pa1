import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# TRAINING CLASS
# =========================
class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(epochs):
            torch.manual_seed(epoch)

            ix = torch.randint(len(data) - context_length, (batch_size,))
            x = torch.stack([data[i:i + context_length] for i in ix])
            y = torch.stack([data[i + 1:i + 1 + context_length] for i in ix])

            logits = model(x)
            B, T, C = logits.shape

            loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 🔥 print loss every epoch
            print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

        return round(loss.item(), 4)


# =========================
# MAIN RUN
# =========================
if __name__ == "__main__":
    print("🚀 Training started...")

    # Import your GPT model
    from model.gpt import GPT

    # Hyperparameters
    vocab_size = 100
    context_length = 10
    epochs = 10
    batch_size = 4
    lr = 1e-3

    # Create model
    model = GPT(vocab_size=vocab_size)

    # Create dummy data
    data = torch.randint(0, vocab_size, (1000,))

    # Train
    sol = Solution()
    final_loss = sol.train(
        model=model,
        data=data,
        epochs=epochs,
        context_length=context_length,
        batch_size=batch_size,
        lr=lr
    )

    print("✅ Training completed!")
    print("Final Loss:", final_loss)
