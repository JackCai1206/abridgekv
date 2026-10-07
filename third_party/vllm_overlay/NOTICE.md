# vLLM overlay notice

This directory contains modified files derived from vLLM 0.13.0.
vLLM is licensed under the Apache License 2.0. The upstream license is included
as `LICENSE`.

The modifications implement CompressLM logical sequencing, cache compaction,
selective-history masks, heterogeneous packed KV storage, and graph-safety
checks. See `README.md` in this directory for the exact patch surface.
