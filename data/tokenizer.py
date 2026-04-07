from typing import List
from collections import Counter

class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # Step 1: split corpus into tokens
        tokens = list(corpus)
        merges = []

        for _ in range(num_merges):
            # Step 2a: count adjacent pairs
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
            freq = Counter(pairs)

            # Step 2b: pick most frequent, break ties lexicographically
            best_pair = min(freq.items(), key=lambda x: (-x[1], x[0]))
            a, b = best_pair[0]
            merges.append([a, b])

            # Step 2c: merge left-to-right, non-overlapping
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens)-1 and tokens[i] == a and tokens[i+1] == b:
                    new_tokens.append(a+b)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # Step 3: return list of merges performed
        return merges

