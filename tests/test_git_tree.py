import subprocess
from pathlib import Path

from sheaf_verify.git_tree import git_tree_sha1


def test_git_tree_matches_git(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "model.py").write_text("class Model:\n    pass\n", encoding="utf-8")
    (repo / "pkg" / "layer.py").write_text("def layer(tokens):\n    return tokens\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "add", "model.py", "pkg/layer.py"], cwd=repo, check=True, capture_output=True)
    tree = subprocess.run(["git", "write-tree"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
    files = []
    for path in (repo / "model.py", repo / "pkg" / "layer.py"):
        files.append((path.relative_to(repo).as_posix(), "100644", path.read_bytes()))
    assert git_tree_sha1(files) == tree
