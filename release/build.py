"""Assemble the Windows distribution from the built onedir shell and a new wheel."""

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import tomllib
from urllib.request import urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release"
UV_URL = ("https://files.pythonhosted.org/packages/69/1b/f40b1ad51226c2ae8707aa124f3b31f9b176a00482048dadd42d9b953d1e/"
          "uv-0.12.13-py3-none-win_amd64.whl")
WEBVIEW_URL = ("https://api.nuget.org/v3-flatcontainer/microsoft.web.webview2/1.0.3856.49/"
               "microsoft.web.webview2.1.0.3856.49.nupkg")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_archive(url, sha):
    path = ROOT / ".cache" / sha
    path.parent.mkdir(exist_ok=True)
    if not path.is_file() or digest(path) != sha:
        with urlopen(url, timeout=120) as response, path.open("wb") as stream:
            shutil.copyfileobj(response, stream)
    if digest(path) != sha:
        raise ValueError(f"Distribution input digest mismatch for {url}")
    return zipfile.ZipFile(path)


def third_party(stage):
    notices = stage / "third-party"
    shutil.copytree(RELEASE / "notices", notices)
    shutil.copy(Path(sys.base_prefix) / "LICENSE.txt", notices / "CPython-LICENSE.txt")
    shutil.copy(Path(sys.base_prefix) / "tcl/tk8.6/license.terms", notices / "Tk-license.terms")
    for dist in metadata.distributions():
        for item in dist.files or ():
            name = item.name.lower()
            if not (name.startswith(("license", "copying", "notice")) and not name.endswith((".py", ".pyc"))):
                continue
            if ".." in item.parts:
                continue
            destination = notices / dist.metadata["Name"] / item
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dist.locate_file(item), destination)
    for package in ("react", "react-dom", "scheduler"):
        shutil.copy(ROOT / "desktop/node_modules" / package / "LICENSE", notices / (package + "-LICENSE"))
    with verified_archive(UV_URL, "3efae0a910a25e18732f38ac1ef7b02e23f47b7fe67735039921fceff01d518c") as bundle:
        (stage / "tools").mkdir()
        for name in bundle.namelist():
            filename = Path(name).name
            if filename == "uv.exe":
                (stage / "tools/uv.exe").write_bytes(bundle.read(name))
            elif "/licenses/" in name and filename.startswith("LICENSE"):
                (notices / ("uv-" + filename)).write_bytes(bundle.read(name))
    with verified_archive(WEBVIEW_URL, "bc0f76eb911b569838dc4aa8f8d325269b966bedb592863d26211aef3a099f1a") as bundle:
        for name in ("LICENSE.txt", "NOTICE.txt"):
            (notices / ("WebView2-SDK-" + name)).write_bytes(bundle.read(name))


def build(out):
    version = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]["version"]
    stem = f"LayerDelta-{version}-windows-x64"
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out) as temp:
        stage = Path(temp) / stem
        shutil.copytree(ROOT / "desktop/build/package/LayerDelta", stage)
        shutil.copytree(RELEASE / "profiles", stage / "profiles")
        for name in ("install.ps1", "Install CPU.cmd", "Install CUDA.cmd"):
            shutil.copyfile(RELEASE / name, stage / name)
        for name in ("LICENSE", "README.md"):
            shutil.copyfile(ROOT / name, stage / name)
        shutil.copytree(ROOT / "docs", stage / "docs",
                        ignore=shutil.ignore_patterns("m4-package.md", "m4-validation.md"))
        shutil.copytree(ROOT / "examples", stage / "examples", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copyfile(ROOT / "agent.md", stage / "agent.md")
        third_party(stage)
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                        "--wheel-dir", str(stage / "wheels"), str(ROOT)], check=True)
        wheel, = (stage / "wheels").glob("*.whl")
        files = {p.relative_to(stage).as_posix(): digest(p) for p in sorted(stage.rglob("*")) if p.is_file()}
        manifest = {"version": version, "wheel": wheel.relative_to(stage).as_posix(), "files": files}
        (stage / "package.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        archive = shutil.make_archive(str(Path(temp) / stem), "zip", temp, stem)
        zip_path = out / (stem + ".zip")
        Path(archive).replace(zip_path)
        sha = digest(zip_path)
        (out / (stem + ".zip.sha256")).write_text(sha + "  " + zip_path.name + "\n", encoding="ascii")
        return {"zip": str(zip_path), "bytes": zip_path.stat().st_size, "sha256": sha, "wheel_sha256": digest(wheel)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    print(json.dumps(build(args.out.resolve())))
