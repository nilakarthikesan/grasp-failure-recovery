"""Explain the frozen linear score without fitting or modifying the model."""
from pathlib import Path
import hashlib
import json
import numpy as np
from grasp_failure_prediction.prediction.dataset import load_dataset, _prefix
from grasp_failure_prediction.prediction.model import load_model, summarize_prefix

source = Path(__file__).resolve().parents[2]
run = Path(__file__).resolve().parent
model_path = source / 'research_artifacts/task_outcome_baseline_v1/proprioception_and_commands.npz'
dataset_dir = source / 'research_artifacts/object_scene_pilot_v1'
report_path = model_path.parent / 'report.json'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
report = json.loads(report_path.read_text())
assert sha(model_path) == report['models']['proprioception_and_commands']['model_sha256']
model = load_model(model_path)
bundle = load_dataset(dataset_dir)
prefix, names, cutoff = _prefix(run / 'prediction_input')
assert names == report['input_contract']['joint_names']
x = summarize_prefix(prefix[None, :])[0]
z = (x - model.normalizer.mean) / model.normalizer.scale
active = model.normalizer.active
weights = np.zeros(len(x))
weights[active] = model.weights
contribution = z * weights
spec = [(stat, field, name) for stat in ('last', 'mean', 'population_std', 'last_minus_first')
        for field in ('qpos', 'qvel', 'command') for name in names]
wrist = np.array(['WRJ' in name for _, _, name in spec])
train_ids = bundle.splits['partitions']['train']['episode_ids']
train = summarize_prefix(bundle.X[np.isin(bundle.episode_ids, train_ids)])
train_min, train_max = train.min(axis=0), train.max(axis=0)
outside = ((x < train_min) | (x > train_max)) & active
p_success = float(model.predict_success(x[None, :])[0])
saved_prediction = json.loads((run / 'prediction.json').read_text())
assert p_success == saved_prediction['p_success']
rows = []
for i in np.argsort(np.abs(contribution))[::-1][:18]:
    stat, field, name = spec[i]
    rows.append(dict(feature_index=int(i), statistic=stat, field=field, joint=name,
                     value=float(x[i]), training_mean=float(model.normalizer.mean[i]),
                     training_scale=float(model.normalizer.scale[i]), standardized_value=float(z[i]),
                     coefficient=float(weights[i]), success_logit_contribution=float(contribution[i]),
                     training_min=float(train_min[i]), training_max=float(train_max[i]),
                     outside_training_range=bool(outside[i])))
result = dict(schema_version='observational_logistic_feature_attribution_v1',
    scope='Attribution of one frozen classifier score; not physical causation or a modified-model result',
    source_sha256={str(path): sha(path) for path in (model_path, report_path,
        dataset_dir/'dataset.npz', dataset_dir/'manifest.json', dataset_dir/'splits.json',
        run/'prediction_input/trajectory.npz', run/'prediction_input/actions.npz')},
    observed_through_s=cutoff, p_success=p_success, p_failure=1-p_success,
    intercept=model.intercept, success_logit=float(model.intercept + sum(contribution)),
    wrist_net_contribution=float(sum(contribution[wrist])),
    nonwrist_net_contribution=float(sum(contribution[~wrist])),
    wrist_absolute_contribution=float(sum(abs(contribution[wrist]))),
    all_absolute_contribution=float(sum(abs(contribution))),
    wrist_absolute_contribution_fraction=float(sum(abs(contribution[wrist])) / sum(abs(contribution))),
    active_feature_count=int(active.sum()), outside_training_range_count=int(outside.sum()),
    wrist_outside_training_range_count=int((outside & wrist).sum()),
    active_abs_z_above_5_count=int(((abs(z)>5) & active).sum()),
    wrist_abs_z_above_5_count=int(((abs(z)>5) & active & wrist).sum()),
    max_active_abs_z=float(max(abs(z[active]))), top_features=rows)
print(json.dumps(result, indent=2))
