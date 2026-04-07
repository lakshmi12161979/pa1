import torch
from typing import List, Tuple

class Solution:
    def batch_loader(
        self,
        raw_dataset: str,
        context_length: int,
        batch_size: int,
        mode: str = "efficient"  # "efficient" or "bindass"
    ) -> Tuple[List[List[str]], List[List[str]]]:
        # Step 1: tokenize
        tokens = raw_dataset.split()

        # Step 2: reproducible random starts
        torch.manual_seed(0)
        starts = torch.randint(
            low=0,
            high=len(tokens) - context_length,
            size=(batch_size,)
        )

        # Step 3: build X and Y
        X = [tokens[s:s+context_length] for s in starts]
        Y = [tokens[s+1:s+1+context_length] for s in starts]

        # Step 4: Pa1 special trace output
        if mode == "bindass":
            print("🔥 Bindass Mode Activated 🔥")
            for i, s in enumerate(starts):
                trace = " → ".join(tokens[s:s+context_length+1])
                print(f"Batch {i+1}:")
                print(f"   X = {X[i]}")
                print(f"   Y = {Y[i]} (shifted right)")
                print(f"   Trace: {trace}")

        return X, Y
