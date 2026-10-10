from pathlib import Path
import subprocess

from grasp_failure_prediction.evaluation import artifacts


def test_source_commit_does_not_depend_on_collection_working_directory(tmp_path, monkeypatch):
    source_root = Path(artifacts.__file__).resolve().parents[3]
    expected = subprocess.check_output(
        ['git', '-C', str(source_root), 'rev-parse', 'HEAD'], text=True,
    ).strip()
    monkeypatch.chdir(tmp_path)
    assert artifacts._code_commit() == expected
