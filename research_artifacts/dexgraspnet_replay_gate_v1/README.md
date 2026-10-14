# DexGraspNet replay gate v1

This artifact records the first ten local evaluations of the official bundled
DexGraspNet mug sample through the provider-neutral MuJoCo evaluator. It contains
only hashes, translation residuals, statuses, and outcome metrics. The CC BY-NC
record and mesh remain in ignored local storage and are not redistributed here.

All cases use `dexgraspnet_saved_record_v1`,
`dexgraspnet_shadow_to_shadow_hand_right_v1`,
`adroit_shadow_tabletop_v2`, and `fixed_grasp_lift_v2`. Seven cases completed as
`approach_collision`, one completed as `failed_acquisition`, and two were
rejected before rollout for canonical Shadow joint-limit violations. Rejections
have no task-success label.
