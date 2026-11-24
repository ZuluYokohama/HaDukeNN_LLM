import subprocess
from pathlib import Path

from sheaf_verify.git_tree import git_tree_sha1


def test_git_tree_matches_git(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    # Write LF bytes and stop git from converting them, so the test holds on Windows too.
    (repo / "model.py").write_bytes(b"class Model:\n    pass\n")
    (repo / "pkg" / "layer.py").write_bytes(b"def layer(tokens):\n    return tokens\n")
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "core.autocrlf=false", "add", "model.py", "pkg/layer.py"], cwd=repo, check=True, capture_output=True
    )
    tree = subprocess.run(["git", "write-tree"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
    files = []
    for path in (repo / "model.py", repo / "pkg" / "layer.py"):
        files.append((path.relative_to(repo).as_posix(), "100644", path.read_bytes()))
    assert git_tree_sha1(files) == tree
