#!/usr/bin/env python3
"""Pipeline runner for HistoriCon setup stages.

Stages (in order):
  1. download    — Patreon RSS → audio in inputs/                (needs PATREON_RSS_TOKEN)
  2. transcribe  — Audio → Greek transcripts with Deepgram        (needs DEEPGRAM_API_KEY)
  3. preprocess  — Clean speaker labels / headers / intro music
  4. embeddings  — Index processed transcripts into ChromaDB

Usage:
    uv run ./run_pipeline.py                       # all stages
    uv run ./run_pipeline.py --from-stage preprocess   # skip download + transcribe
    uv run ./run_pipeline.py --only embeddings         # just one stage
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).parent


@dataclass(frozen=True)
class Stage:
    name: str
    script: str
    description: str
    required_env: tuple[str, ...] = ()


STAGES: tuple[Stage, ...] = (
    Stage("download", "download_patreon.py",
          "Download audio from Patreon RSS", ("PATREON_RSS_TOKEN",)),
    Stage("transcribe", "transcribe_deepgram.py",
          "Transcribe audio to Greek text with Deepgram", ("DEEPGRAM_API_KEY",)),
    Stage("preprocess", "preprocess_transcripts.py",
          "Clean and format transcripts"),
    Stage("embeddings", "create_embeddings.py",
          "Create embeddings and index transcripts"),
)


def preflight(stages: list[Stage]) -> list[str]:
    """Return a list of missing env vars across the selected stages."""
    missing = []
    for stage in stages:
        for var in stage.required_env:
            if not os.getenv(var):
                missing.append(f"{var} (required by stage '{stage.name}')")
    return missing


def run_stage(stage: Stage) -> bool:
    script_path = REPO_ROOT / "scripts" / stage.script
    print(f"\n{'=' * 80}\nStage: {stage.name} ({stage.script})\n{stage.description}\n{'=' * 80}\n")
    try:
        subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=REPO_ROOT,
        )
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Stage '{stage.name}' failed (exit {e.returncode})")
        return False
    except Exception as e:  # noqa: BLE001 — top-level orchestrator catch
        print(f"\n❌ Stage '{stage.name}' failed: {e}")
        return False
    print(f"\n✅ Stage '{stage.name}' completed")
    return True


def select_stages(args: argparse.Namespace) -> list[Stage]:
    names = [s.name for s in STAGES]
    if args.only:
        return [s for s in STAGES if s.name == args.only]
    start = names.index(args.from_stage) if args.from_stage else 0
    return list(STAGES[start:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    names = [s.name for s in STAGES]
    parser.add_argument("--from-stage", choices=names, help="Start at this stage (skip earlier ones)")
    parser.add_argument("--only", choices=names, help="Run only this stage")
    args = parser.parse_args()

    if args.only and args.from_stage:
        parser.error("--only and --from-stage are mutually exclusive")

    selected = select_stages(args)
    print("HistoriCon Pipeline Runner")
    print(f"Stages to run: {', '.join(s.name for s in selected)}\n")

    missing = preflight(selected)
    if missing:
        print("❌ Missing required environment variables:")
        for m in missing:
            print(f"   • {m}")
        sys.exit(2)

    for stage in selected:
        if not run_stage(stage):
            print(f"\n⚠️  Pipeline stopped at '{stage.name}'. Fix and re-run with --from-stage {stage.name}.")
            sys.exit(1)

    print(f"\n{'=' * 80}\n🎉 All selected stages completed.\n{'=' * 80}")
    print("\nReady: uv run python agents/web_orchestrator.py")


if __name__ == "__main__":
    main()
