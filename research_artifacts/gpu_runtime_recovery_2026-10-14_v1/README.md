# A40 runtime recovery, 14 October 2026

The user authorized retrying original pod 8ibam2z5n6ea5r, then using one available A40 if the retry failed, while keeping the original and generating no new eval data. The retry returned the same host-capacity error. The live Secure Cloud catalog showed Medium A40 stock in CA-MTL-1 at $0.59/hour before creation.

Replacement 10fvywuiko09rk is running on an A40 at the recorded $0.59/hour rate. Its immutable PyTorch image is copied from the original pod config, not resolved from a moving tag. Creation's 55-second SSH wait expired while the image initialized; the existing created ID was inspected and allowed to finish. No second replacement was created. SSH subsequently bound and authenticated.

Both backups matched their existing local SHA256 before copying and again remotely before extraction. Empty destinations were required. The restore checked 20,760 runtime regular files, 319 asset regular files and three symlinks against archive contents. It did not overwrite the original pod. Runtime imports, pip check, source commit/input/checkpoint hashes and a fixed 2x2 CUDA multiplication all pass. Checkpoint loading, generator imports and physics imports are prohibited by the preflight. The restored Python is 3.8.12, Torch 1.11.0/CUDA 11.3. This does not run or certify DGA inference or HUG mapping.

Runtime: /opt/dga/runtime_py38_fast; editable dependency sources: /opt/dga/deps. Original source, checkpoint and conditioning: /workspace/dga. Verified backup archives and recovery scripts/reports are retained in /workspace/grasp-runtime-recovery. No licensed hand assets or backup archives are added to Git.

The original pod stays stopped and retains its volume. The new pod keeps billing while running, including while idle; retained disk also bills. The original stopped disk is still retained and billed. /workspace is the new pod's 20 GB volume, not a separately created network volume. On stop, /opt is cleared; restore the saved runtime overlay onto the same pinned image after restart. SSH addresses can change on restart. Read the detailed project recovery note before replaying scripts: they deliberately refuse existing restore destinations and report paths.

This is an operational execution record, not a scientific preregistration or new evaluation dataset. Historical generator and mapping results are unchanged.
