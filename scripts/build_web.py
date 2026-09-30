"""Build the browser app into build/web, and optionally serve it.

    python scripts/build_web.py            # write build/web
    python scripts/build_web.py --serve    # build, then serve http://localhost:8000

The site is web/ plus a zip of the anasim package. Pyodide, numpy, and scipy
load from the Pyodide CDN.
"""

import argparse
import functools
import hashlib
import http.server
import io
import json
import re
import shutil
import zipfile
from pathlib import Path

PYODIDE_VERSION = "314.0.7"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "build" / "web"


def package_zip() -> bytes:
    """Zip anasim/*.py with fixed timestamps so the name changes only with content."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((ROOT / "anasim").rglob("*.py")):
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


def check_output_dir(out: Path) -> None:
    """Refuse outputs whose replacement would delete sources or unrelated files."""
    source = ROOT / "web"
    if out == source or source in out.parents:
        raise SystemExit(f"Refusing to build into {out}: it is inside the web/ sources.")
    if out.exists() and any(out.iterdir()) and not (out / "build.json").is_file():
        raise SystemExit(f"Refusing to replace {out}: it is not empty and is not a previous build.")


def build(out: Path) -> None:
    check_output_dir(out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "web", out)
    data = package_zip()
    name = f"anasim-{hashlib.sha256(data).hexdigest()[:12]}.zip"
    (out / name).write_bytes(data)
    version = re.search(r'^__version__ = "([^"]+)"$', (ROOT / "anasim" / "__init__.py").read_text(), re.M)
    (out / "build.json").write_text(json.dumps({
        "version": version.group(1),
        "package": name,
        "pyodide": f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/",
    }, indent=2) + "\n")
    # GitHub Pages would otherwise run Jekyll over the site.
    (out / ".nojekyll").touch()
    print(f"Built {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out} ({len(data) // 1024} KB package)")


def serve(out: Path, port: int) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        print(f"Serving http://localhost:{port}/ (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    parser.add_argument("--serve", action="store_true", help="serve the build after writing it")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--pyodide-version", action="store_true", help="print the Pyodide version and exit")
    args = parser.parse_args()
    if args.pyodide_version:
        print(PYODIDE_VERSION)
        return
    build(args.out.resolve())
    if args.serve:
        serve(args.out.resolve(), args.port)


if __name__ == "__main__":
    main()
