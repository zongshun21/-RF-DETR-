# Model checkpoint chunks

The two best checkpoints exceed GitHub's 100 MB per-file limit. Each file is
split into ordered 8 MiB parts. From the repository root, run:

```bash
bash scripts/download_release_weights.sh
```

This concatenates the parts into `weights/releases/` and verifies the original
files against `SHA256SUMS`. Part names use fixed-width numeric suffixes, so shell
lexicographic order is checkpoint byte order.
