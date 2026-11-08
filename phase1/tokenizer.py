# BPE tokenizer from Assignment 1 (word-frequency training, cached encoding)
import json
import os
import time
from collections import defaultdict


class BPETokenizer:

    def __init__(self, vocab_size=16384):
        self.target_vocab_size = vocab_size
        self.itos = ["<pad>", "<unk>", "<s>", "</s>"]
        self.itos.extend(range(256))
        self.eow_id = len(self.itos)
        self.itos.append("</w>")
        self.merges = []
        self.vocab_size = len(self.itos)
        self._merge_rank = {}
        self._encode_cache = {}
        self._token_bytes = None
        self._ends_word = None

    def fit(self, corpus, checkpoint_dir=None, checkpoint_every=1000):
        ckpt_path = None
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir, "bpe_v" + str(self.target_vocab_size) + ".ckpt.json")

        resumed_merge = 0
        word_data = None

        if ckpt_path and os.path.exists(ckpt_path):
            print("Found checkpoint at", ckpt_path, "resuming...")
            with open(ckpt_path, "r") as f:
                ckpt = json.load(f)
            self.merges = [((a, b), c) for (a, b), c in ckpt["merges"]]
            self.itos = ["<pad>", "<unk>", "<s>", "</s>"]
            self.itos.extend(range(256))
            self.itos.append("</w>")
            for pair, new_id in self.merges:
                self.itos.append(pair)
            self.vocab_size = len(self.itos)
            resumed_merge = len(self.merges)
            word_data = []
            for tokens, freq in ckpt["words"]:
                word_data.append([tokens, freq])
            print(f"  Resumed after {resumed_merge} merges (vocab: {self.vocab_size})")

        if word_data is None:
            print("Counting word frequencies...")
            word_freq = {}
            for word in corpus.split():
                word_freq[word] = word_freq.get(word, 0) + 1
            print(f"  {len(word_freq)} unique words")

            word_data = []
            for word, freq in word_freq.items():
                tokens = [b + 4 for b in word.encode("utf-8")]
                tokens.append(self.eow_id)
                word_data.append([tokens, freq])

        print("Building pair index...")
        pair_counts = defaultdict(int)
        pair_words = defaultdict(set)

        for idx, (tokens, freq) in enumerate(word_data):
            seen = defaultdict(int)
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                seen[pair] += 1
            for pair, cnt in seen.items():
                pair_counts[pair] += cnt * freq
                pair_words[pair].add(idx)

        num_merges_total = self.target_vocab_size - 261  # 4 specials + 256 bytes + </w>
        num_merges_left = num_merges_total - resumed_merge
        print(f"Learning {num_merges_left} merges (of {num_merges_total} total)...")
        t0 = time.time()

        for merge_i in range(num_merges_left):
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 2:
                break

            new_id = len(self.itos)
            bp_a, bp_b = best_pair

            affected = list(pair_words.get(best_pair, set()))

            for idx in affected:
                tokens, freq = word_data[idx]

                old_pairs = defaultdict(int)
                for i in range(len(tokens) - 1):
                    old_pairs[(tokens[i], tokens[i + 1])] += 1
                for p, cnt in old_pairs.items():
                    pair_counts[p] -= cnt * freq
                    if pair_counts[p] <= 0:
                        pair_counts.pop(p, None)
                    pair_words[p].discard(idx)
                    if not pair_words[p]:
                        pair_words.pop(p, None)

                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == bp_a and tokens[i + 1] == bp_b:
                        new_tokens.append(new_id)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                word_data[idx][0] = new_tokens

                new_pairs = defaultdict(int)
                for i in range(len(new_tokens) - 1):
                    new_pairs[(new_tokens[i], new_tokens[i + 1])] += 1
                for p, cnt in new_pairs.items():
                    pair_counts[p] = pair_counts.get(p, 0) + cnt * freq
                    pair_words[p].add(idx)

            self.merges.append((best_pair, new_id))
            self.itos.append(best_pair)
            self.vocab_size = len(self.itos)

            done = resumed_merge + merge_i + 1
            if done % 1000 == 0:
                elapsed = time.time() - t0
                rate = (merge_i + 1) / elapsed
                if rate > 0:
                    eta = (num_merges_left - merge_i - 1) / rate
                else:
                    eta = 0
                print(f"  {done}/{num_merges_total} merges | {elapsed:.0f}s elapsed | ~{eta / 60:.1f} min remaining")

            if ckpt_path and (merge_i + 1) % checkpoint_every == 0:
                self._save_checkpoint(ckpt_path, word_data)

        if ckpt_path:
            self._save_checkpoint(ckpt_path, word_data)

        self._rebuild_caches()
        total_time = time.time() - t0
        print(f"Done. Final vocab: {self.vocab_size} | Time: {total_time / 60:.1f} min")
        return self

    def _save_checkpoint(self, path, word_data):
        ckpt = {
            "merges": [[[a, b], c] for (a, b), c in self.merges],
            "words": [[tokens, freq] for tokens, freq in word_data],
        }
        with open(path, "w") as f:
            json.dump(ckpt, f)

    def _rebuild_caches(self):
        self._merge_rank = {}
        for rank, ((a, b), new_id) in enumerate(self.merges):
            self._merge_rank[(a, b)] = (rank, new_id)
        self._encode_cache = {}

        self._token_bytes = {}
        self._ends_word = {}
        for x in range(256):
            self._token_bytes[x + 4] = bytes([x])
            self._ends_word[x + 4] = False
        self._token_bytes[self.eow_id] = b""
        self._ends_word[self.eow_id] = True
        for (a, b), c in self.merges:
            self._token_bytes[c] = self._token_bytes[a] + self._token_bytes[b]
            self._ends_word[c] = self._ends_word[b]

    def _encode_word(self, word):
        tokens = [b + 4 for b in word.encode("utf-8")]
        tokens.append(self.eow_id)

        while len(tokens) >= 2:
            best_rank = float("inf")
            best_idx = -1
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                r = self._merge_rank.get(pair)
                if r is not None and r[0] < best_rank:
                    best_rank = r[0]
                    best_idx = i

            if best_idx == -1:
                break

            _, new_id = self._merge_rank[(tokens[best_idx], tokens[best_idx + 1])]
            tokens = tokens[:best_idx] + [new_id] + tokens[best_idx + 2:]

        return tokens

    def encode(self, text):
        if not self._merge_rank and self.merges:
            self._rebuild_caches()

        result = []
        for word in text.split():
            cached = self._encode_cache.get(word)
            if cached is not None:
                result.extend(cached)
            else:
                tokens = self._encode_word(word)
                self._encode_cache[word] = tokens
                result.extend(tokens)
        return result

    def decode(self, tokens):
        if self._token_bytes is None:
            self._rebuild_caches()

        parts = []
        for i, token in enumerate(tokens):
            b = self._token_bytes.get(token)
            if b is not None:
                parts.append(b)
                if self._ends_word.get(token, False) and i < len(tokens) - 1:
                    parts.append(b" ")

        return b"".join(parts).decode("utf-8", errors="replace")

    def save(self, path):
        data = {
            "vocab_size": self.vocab_size,
            "target_vocab_size": self.target_vocab_size,
            "merges": [[[a, b], c] for (a, b), c in self.merges],
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Tokenizer saved to {path} ({self.vocab_size} tokens)")

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        tok = cls(data["target_vocab_size"])
        tok.merges = [((a, b), c) for (a, b), c in data["merges"]]
        for pair, new_id in tok.merges:
            tok.itos.append(pair)
        tok.vocab_size = len(tok.itos)
        tok._rebuild_caches()
        return tok
