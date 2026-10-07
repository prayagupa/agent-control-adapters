"""Check package publication metadata and fail-closed manual release policy."""

from __future__ import annotations

import os
import re
import subprocess
import tomllib
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml
from agent_control_adapters.json import object_value

ROOT: Path = Path(__file__).resolve().parents[1]
REPOSITORY: str = "https://github.com/prayagupa/agent-control-adapters"
MANIFESTS: tuple[Path, ...] = tuple(sorted((ROOT / "python" / "packages").glob("*/pyproject.toml")))


@pytest.mark.parametrize("manifest", MANIFESTS, ids=[path.parent.name for path in MANIFESTS])
def test_product_distributions_and_namespaces_use_agent_control(manifest: Path) -> None:
    """Keep product names aligned without renaming the external Agent Hooks SDK."""
    metadata = object_value(tomllib.loads(manifest.read_text(encoding="utf-8")))
    project = object_value(metadata["project"])
    suffix = (
        "adapters" if manifest.parent.name == "core" else manifest.parent.name.replace("-", "_")
    )
    namespace = f"agent_control_{suffix}"
    assert project["name"] == namespace.replace("_", "-")
    tool = object_value(metadata["tool"])
    hatch = object_value(tool["hatch"])
    build = object_value(hatch["build"])
    targets = object_value(build["targets"])
    wheel = object_value(targets["wheel"])
    assert wheel["packages"] == [f"src/{namespace}"]
    assert (manifest.parent / "src" / namespace / "__init__.py").is_file()
    assert (manifest.parent / "src" / namespace / "py.typed").is_file()
    assert not (manifest.parent / "src" / f"agent_hooks_{suffix}").exists()
    dependencies = project["dependencies"]
    assert isinstance(dependencies, list)
    if suffix == "adapters":
        assert dependencies == ["agent-hooks-sdk==0.1.0a5"]
    else:
        assert "agent-control-adapters==0.1.0a1" in dependencies


@pytest.mark.parametrize("manifest", MANIFESTS, ids=[path.parent.name for path in MANIFESTS])
def test_package_metadata_is_explicit_and_links_to_shipped_docs(manifest: Path) -> None:
    """Keep alpha status, licensing and canonical documentation in each distribution."""
    metadata = object_value(tomllib.loads(manifest.read_text(encoding="utf-8")))
    project = object_value(metadata["project"])
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]
    classifiers = project["classifiers"]
    assert isinstance(classifiers, list)
    assert "Development Status :: 3 - Alpha" in classifiers
    assert "Typing :: Typed" in classifiers
    urls = object_value(project["urls"])
    assert urls["Homepage"] == REPOSITORY
    assert urls["Source"] == REPOSITORY
    assert urls["Issues"] == f"{REPOSITORY}/issues"
    documentation = urls["Documentation"]
    assert isinstance(documentation, str)
    prefix = f"{REPOSITORY}/blob/main/"
    assert documentation.startswith(prefix)
    assert (ROOT / documentation.removeprefix(prefix)).is_file()


@pytest.mark.parametrize("manifest", MANIFESTS, ids=[path.parent.name for path in MANIFESTS])
def test_each_package_carries_the_original_license(manifest: Path) -> None:
    """Ship the complete existing notice without changing its ownership."""
    assert (manifest.parent / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()


@pytest.mark.parametrize("manifest", MANIFESTS, ids=[path.parent.name for path in MANIFESTS])
def test_package_readmes_are_short_and_portable(manifest: Path) -> None:
    """Keep descriptions brief and links usable outside a source checkout."""
    readme = (manifest.parent / "README.md").read_text(encoding="utf-8")
    assert len(readme.splitlines()) <= 25
    assert "alpha" in readme.lower()
    links = re.findall(r"\]\(([^)]+)\)", readme)
    assert links
    assert all(link.startswith("https://") for link in links)
    assert any(link.startswith(f"{REPOSITORY}/blob/main/docs/") for link in links)


def release_workflow() -> Mapping[str, object]:
    """Parse workflow policy as data, without dispatching any GitHub operation."""
    return object_value(yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text()))


def release_script(step_id: str) -> str:
    """Select exactly one shell step from the real build workflow."""
    jobs = object_value(release_workflow()["jobs"])
    steps = object_value(jobs["build"])["steps"]
    assert isinstance(steps, list)
    matches = [object_value(step) for step in steps if object_value(step).get("id") == step_id]
    assert len(matches) == 1
    script = matches[0]["run"]
    assert isinstance(script, str)
    return script


def test_release_is_manual_and_dry_run_by_default() -> None:
    """Pushes, tags and releases must never trigger publication automatically."""
    workflow = release_workflow()
    events = object_value(workflow["on"])
    assert set(events) == {"workflow_dispatch"}
    dispatch = object_value(events["workflow_dispatch"])
    inputs = object_value(dispatch["inputs"])
    dry_run = object_value(inputs["dry_run"])
    assert dry_run == {
        "description": "Build all Python packages without attestation or PyPI publication",
        "type": "boolean",
        "required": True,
        "default": True,
    }
    assert workflow["permissions"] == {"contents": "read"}
    assert object_value(workflow["concurrency"])["cancel-in-progress"] is False


def test_publication_is_guarded_and_actions_are_pinned() -> None:
    """Limit publishing credentials to an explicitly non-dry protected job."""
    jobs = object_value(release_workflow()["jobs"])
    assert set(jobs) == {"build", "publish"}
    build = object_value(jobs["build"])
    assert build["permissions"] == {"contents": "read", "actions": "read"}
    publish = object_value(jobs["publish"])
    assert publish["if"] == "${{ github.event_name == 'workflow_dispatch' && !inputs.dry_run }}"
    assert publish["needs"] == "build"
    assert publish["environment"] == "release"
    assert publish["permissions"] == {
        "contents": "read",
        "id-token": "write",
        "attestations": "write",
    }
    for value in jobs.values():
        steps = object_value(value)["steps"]
        assert isinstance(steps, list)
        for step in steps:
            action = object_value(step).get("uses")
            if action is not None:
                assert isinstance(action, str)
                assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", action)


@pytest.mark.parametrize(
    ("dry_run", "ref_type", "ref_name", "ci_state", "mismatch", "expected"),
    [
        ("true", "branch", "main", "", False, 0),
        ("true", "branch", "main", "", True, 1),
        ("false", "branch", "main", "completed\tsuccess", False, 1),
        ("false", "tag", "v0.2.0", "completed\tsuccess", False, 1),
        ("false", "tag", "v0.1.0a1", "", False, 1),
        ("false", "tag", "v0.1.0a1", "in_progress\t", False, 1),
        ("false", "tag", "v0.1.0a1", "completed\tfailure", False, 1),
        ("false", "tag", "v0.1.0a1", "completed\tsuccess", False, 0),
    ],
)
def test_release_prerequisites_fail_closed(
    tmp_path: Path,
    dry_run: str,
    ref_type: str,
    ref_name: str,
    ci_state: str,
    mismatch: bool,
    expected: int,
) -> None:
    """Execute the actual guard with stub CLI boundaries, never publication."""
    script = release_script("metadata")
    uv = tmp_path / "uv"
    uv.write_text(
        '#!/bin/bash\nif [[ "$MISMATCH" == true && "$*" == *"/maf"* ]]; then\n'
        "  echo 0.2.0\nelse\n  echo 0.1.0a1\nfi\n"
    )
    gh = tmp_path / "gh"
    gh.write_text('#!/bin/bash\nprintf "%s\\n" "$CI_STATE"\n')
    uv.chmod(0o700)
    gh.chmod(0o700)
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", script],
        env={
            "PATH": f"{tmp_path}{os.pathsep}{os.defpath}",
            "DRY_RUN": dry_run,
            "GITHUB_REF_TYPE": ref_type,
            "GITHUB_REF_NAME": ref_name,
            "GITHUB_REPOSITORY": "prayagupa/agent-control-adapters",
            "GITHUB_SHA": "0" * 40,
            "GITHUB_OUTPUT": str(tmp_path / "outputs"),
            "CI_STATE": ci_state,
            "MISMATCH": str(mismatch).lower(),
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    if expected == 0:
        assert (tmp_path / "outputs").read_text() == "version=0.1.0a1\n"
    else:
        assert "::error::" in result.stdout
        assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("complete", 0),
        ("missing-wheel", 1),
        ("missing-sdist", 1),
        ("wrong-project", 1),
        ("extra-wheel", 1),
    ],
)
def test_release_requires_exactly_seven_named_distribution_pairs(
    tmp_path: Path, variant: str, expected: int
) -> None:
    """Unit-test inventory rejection with empty fixtures, not released artifacts."""
    packages = tmp_path / "dist/release/packages"
    packages.mkdir(parents=True)
    for manifest in MANIFESTS:
        name = (
            "adapters" if manifest.parent.name == "core" else manifest.parent.name.replace("-", "_")
        )
        prefix = f"agent_control_{name}-0.1.0a1"
        (packages / f"{prefix}-py3-none-any.whl").touch()
        (packages / f"{prefix}.tar.gz").touch()
    core_wheel = packages / "agent_control_adapters-0.1.0a1-py3-none-any.whl"
    if variant == "missing-wheel":
        core_wheel.unlink()
    elif variant == "missing-sdist":
        (packages / "agent_control_adapters-0.1.0a1.tar.gz").unlink()
    elif variant == "wrong-project":
        core_wheel.rename(packages / "wrong-0.1.0a1-py3-none-any.whl")
    elif variant == "extra-wheel":
        (packages / "extra-0.1.0a1-py3-none-any.whl").touch()
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/bash\nexit 0\n")
    uv.chmod(0o700)
    result = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-e",
            "-o",
            "pipefail",
            "-c",
            release_script("distributions"),
        ],
        cwd=tmp_path,
        env={
            "PATH": f"{tmp_path}{os.pathsep}{os.defpath}",
            "RELEASE_VERSION": "0.1.0a1",
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    checksums = tmp_path / "dist/release/SHA256SUMS"
    if expected == 0:
        assert len(checksums.read_text().splitlines()) == 14
    else:
        assert "::error::" in result.stdout
        assert not checksums.exists()
