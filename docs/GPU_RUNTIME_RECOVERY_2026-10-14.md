# Restored A40 development runtime

14 October 2026, America/New_York. The original pod 8ibam2z5n6ea5r still failed its authorized start with insufficient free GPUs on its host. It remains stopped and preserved. A single replacement A40, 10fvywuiko09rk (grasp-validation-a40-20261014), is running in CA-MTL-1 Secure Cloud at the observed $0.59/hour rate, funded by the existing account balance. The live catalog showed Medium stock before creation. Running compute continues billing until stopped; both retained pod disks also bill.

The exact image, read from the original pod, is pytorch/pytorch@sha256:9bfcfa72b6b244c1fbfa24864eec97fb29cfafc065999e9a9ba913fa1e690a02 (the PyTorch 1.11/CUDA 11.3 development image). The new shape is one A40, 30 GB container disk, 20 GB /workspace volume and TCP 22 only. No network volume or additional replacement was created.

Verified connection at handoff:

```bash
ssh -i /Users/nilakarthikesan/.ssh/runpod_codex_dga_ed25519 -p 22086 root@69.30.85.41
```

The key was matched to the existing registered public-key fingerprint before boot. The CLI's default local SSH-key path was absent; the verified account-matching key above authenticated successfully. The IP and port can change on restart; reread live SSH info instead of reusing this address blindly.

The existing snapshot restored /opt/dga/runtime_py38_fast and /opt/dga/deps. Original DGA source, checkpoint and conditioning restored to /workspace/dga. Both transferred archive hashes match the existing verified backups; every restored regular file and symlink matches its archive. A no-model preflight passes Torch 1.11.0/CUDA 11.3, actual A40 compute 8.6, compiled library imports, pip check, pinned source and six asset hashes, unchanged source bytes and a fixed 2x2 CUDA multiplication. No checkpoint was deserialized, generator imported, inference called, physics stepped, eval episode/label created or model trained.

The backups, restore script and preflight are retained under /workspace/grasp-runtime-recovery; evidence copies and exact hashes are in [the recovery record](../research_artifacts/gpu_runtime_recovery_2026-10-14_v1/README.md). The first create wait timed out during image initialization, after successful allocation; the same pod finished booting and passed real SSH verification. It was not treated as a failed allocation or duplicated.

After a stop, /workspace survives but /opt is cleared. On the same pinned image, first verify the saved runtime archive against SHA256 2524a50a6f4bf0533493097de27bd3b0255607df6dfc6c0f9ca5f6d87c9e51de, confirm /opt/dga is absent, then extract only that overlay under /opt. Do not rerun the full archive restore against already-restored /workspace/dga: it deliberately refuses existing destinations. The original preflight report is immutable; use a separately recorded new report path/script version for another check. Local backups remain available if the new pod's disk is ever lost.

This restores the saved DGA runtime. HUG's current static geometry checks run on the existing CPU runtime; no automatic GPU speedup or HUG compatibility is claimed. New generator inference/eval data still requires the appropriate separate authorization and frozen scientific protocol.
