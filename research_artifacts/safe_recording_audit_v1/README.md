# SAFE public recording audit v1

Offline inspection on 12 October 2026. Read the [walkthrough](../../docs/SAFE_RECORDING_WALKTHROUGH.md) for the simple explanation and next evidence gate.

This bundle contains source metadata, verified extracted-member hashes and qualitative observations. It contains no redistributed videos, feature tensors, fitted model, predictions, calibration artifacts or training split. The original project predictor is unchanged.

The official SAFE release links the WidowX archive. We inventoried 532 CSV/PKL/MP4 triplets and selected the first carrot collection in archive order, then its lowest raw success and failure episode IDs before viewing. Six members were extracted with ZIP CRC checks. PKL inspection used passive pickle opcode parsing; no unpickling or Torch loading ran. File hashes are in [manifest.json](manifest.json).

Task 6, Bridge V2, Put the Carrot on Plate: raw episode 3 succeeds and raw episode 0 fails according to source outcomes. Visual inspection shows object transport in episode 3 and an empty raised gripper in episode 0. Their initial object poses differ. [qualitative_observations.json](qualitative_observations.json) separates this raw pair from the authors' Franka demo, which includes alert overlays and eventually warns on a failed task.

Raw local files and the contact sheet are stored at `/Users/nilakarthikesan/grasp-failure-prediction/runs/safe_recording_audit_2026-10-12/`. The complete ZIP was not downloaded; selected compressed member payloads total 5,277,364 bytes. Archive preview, tail and directory metadata were fetched separately. Terms remain unverified; the website template license does not establish dataset rights.

There are no SAFE scores/thresholds in the selected CSVs, and no trained-monitor/calibration artifacts were found in the reviewed public Git release or archive filename inventory. We have not measured SAFE errors on the raw pair. Source outcome labels and Codex visual descriptions are not independently reviewed grasp-event labels. Encoded video seconds are not verified physical event time. This is illustrative evidence, not a novelty claim or a real-world model evaluation.
