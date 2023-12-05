# merkle-cbt-lean
Merkle tree implementation for constrained memory environments

This crate is based on CBMT implementation in https://crates.io/crates/merkle-cbt

Although indeed efficient both in performance and memory requirements, abovementioned crates memory consumption was found to be too high for its use in really constrained embedded environments (like [Kampela device](https://github.com/Kalapaja/kampela-firmware), or Ledger). We have created this crate to handle proof check with as small memory access and consumption as possible, so that proof lemmas could be stored on external memory or tranferred through serial data interface. Note that conctruction of proof is not optimized, as in typical application it is performed in a regular memory environment and its difficulty is negligible. Root value calculated by this crate and `merkle-cbt` is designed and checked to be identical, this above crate should be used instead of this one where memory limitations are not critical.

We've decided to make this a separate crate to avoid confusion in proof structures and to keep both modulse as minimalistic as they are.

## Concept

To limit memory, we've replaced width graph traverse with depth. Thus, only one branch at a time is stored simultaneously, and CMBT is always much wider than it is tall. To achieve this, we've had to change sorting of the lemmas accordingly, otherwise construction is conceptually same as in abovementioned crate. Main drawback here is roughly quadratic traverse time as opposed to linear width traverse, as, starting from below, we have to traverse whole tree for every leaf.

## Detailed operation

### Proof construction

Currently the proof is constructed in a broad way, unused nodes are merged together until no merges are possible; then resulting lemmas are sorted strictly from left to right; this is lemma transfer order.

### Proof checking storage

A buffer with depth corresponding to tree height stores already calculated left node values in current path. This buffer should reside within "internal" faster memory as opposed to where lemmas are located for this project to show performance improvement. This is all storage, apart from path storage itself and space needed for `Merge` procedure, that is used by proof check.

Every time a node is merged into this buffer, it is written into corresponding layer if it is left node, or merged from the right with the value on its level and moved for merge.

### Proof check algorithm

Leftmost leaf is found and its path to root is consructed. All proofs to the left of the new path and to the right of last path are merged into buffer from left to right: first lemmas immediately to the right of last path are merged going from bottom up, then lemmas to the left of new path are merged top-down, leaving nodes above paths intersection point intact. Then the leaf is merged into the buffer. This procedure is repeated for every leaf until all are used up, all remaining proofs on the right of the last path are then merged from bottom up (left to right). Lemmas pool should be checked for total lemmas consumption.

## Notes

Naturally, this crate supports 'no-std'
