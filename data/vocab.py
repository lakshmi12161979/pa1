from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Step 1: collect unique characters
        chars = sorted(set(text))
        
        # Step 2: assign integer IDs alphabetically
        stoi = {ch: idx for idx, ch in enumerate(chars)}
        itos = {idx: ch for ch, idx in stoi.items()}
        
        return stoi, itos

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Step 3: map each character to its integer ID
        return [stoi[ch] for ch in text]

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Step 4: map each integer ID back to its character
        return "".join(itos[idx] for idx in ids)
