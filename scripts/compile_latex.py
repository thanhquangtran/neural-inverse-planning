from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

def _configure_local_b8tex_dirs(project_root: Path) -> None:
    cache_home = project_root / ".cache"
    config_home = project_root / ".config"
    cache_home.mkdir(parents=True, exist_ok=True)
    config_home.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(cache_home)
    os.environ["XDG_CONFIG_HOME"] = str(config_home)


def _collect_resources(root: Path, entrypoint: Path, output_dir: Path) -> list[object]:
    from b8tex import Resource

    resources: list[object] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path == entrypoint:
            continue
        if output_dir in path.parents:
            continue
        resources.append(Resource(path))
    return resources


def build_latex(entrypoint: Path) -> Path:
    project_root = Path(__file__).resolve().parent.parent
    _configure_local_b8tex_dirs(project_root)
    from b8tex import BuildOptions, Document, Resource, TectonicCompiler

    latex_dir = entrypoint.parent.resolve()
    entrypoint = entrypoint.resolve()
    output_dir = latex_dir / "build"
    output_dir.mkdir(parents=True, exist_ok=True)

    document = Document(
        name=entrypoint.stem,
        entrypoint=entrypoint,
        resources=_collect_resources(latex_dir, entrypoint, output_dir),
    )

    compiler = TectonicCompiler(use_cache=False)
    compiler.compile_document(
        document=document,
        options=BuildOptions(outdir=output_dir),
    )
    pdf_path = output_dir / f"{entrypoint.stem}.pdf"
    if not pdf_path.exists():
        raise RuntimeError(f"expected PDF output at {pdf_path}, but it was not created")
    return pdf_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile the LaTeX report with b8tex/tectonic.")
    parser.add_argument(
        "entrypoint",
        nargs="?",
        default="latex/main.tex",
        help="path to the main .tex file (default: latex/main.tex)",
    )
    args = parser.parse_args()

    entrypoint = Path(args.entrypoint)
    if not entrypoint.exists():
        parser.error(f"missing LaTeX entrypoint: {entrypoint}")
    if entrypoint.suffix != ".tex":
        parser.error(f"expected a .tex file, got: {entrypoint}")

    try:
        pdf_path = build_latex(entrypoint)
    except Exception as exc:
        try:
            from b8tex import CompilationFailed
        except Exception:
            CompilationFailed = None
        if CompilationFailed is not None and isinstance(exc, CompilationFailed):
            for error in exc.errors:
                print(error, file=sys.stderr)
            if exc.log_tail:
                print(exc.log_tail, file=sys.stderr)
            return 1
        raise

    print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
