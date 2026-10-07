"""Check portable package descriptions and independently distributable licensing."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
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
