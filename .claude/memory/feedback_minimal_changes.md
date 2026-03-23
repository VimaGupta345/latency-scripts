---
name: minimal_changes_only
description: When fixing a bug or changing one thing, do NOT change unrelated defaults or parameters. Only touch what's broken.
type: feedback
---

When fixing an issue (e.g., wrong benchmark task name), ONLY change the specific broken thing. Do not change other parameters like model paths, defaults, or configurations that were working fine. The user was frustrated when a benchmark name fix also silently changed the default model from DeepSeek-R1 to DeepSeek-V3.
