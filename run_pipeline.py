#!/usr/bin/env python3
"""
Pipeline runner for HistoriCon setup stages.

Runs all setup scripts in order:
1. download_patreon.py - Download audio from Patreon RSS
2. transcribe_deepgram.py - Transcribe audio to Greek text
3. preprocess_transcripts.py - Clean and format transcripts
4. create_embeddings.py - Index transcripts for RAG queries
"""

import subprocess
import sys
from pathlib import Path


def run_script(script_name: str, description: str) -> bool:
    """Run a setup script and report status.
    
    Args:
        script_name: Name of the script file (e.g., "download_patreon.py")
        description: Human-readable description of what the script does
        
    Returns:
        True if script succeeded, False otherwise
    """
    script_path = Path(__file__).parent / "scripts" / script_name
    
    print(f"\n{'=' * 80}")
    print(f"Running: {script_name}")
    print(f"Description: {description}")
    print(f"{'=' * 80}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=Path(__file__).parent
        )
        print(f"\n✅ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {script_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ {script_name} failed with error: {e}")
        return False


def main():
    """Run all setup scripts in sequence."""
    print("HistoriCon Pipeline Runner")
    print("Running all setup stages in order...\n")
    
    scripts = [
        ("download_patreon.py", "Download audio from Patreon RSS"),
        ("transcribe_deepgram.py", "Transcribe audio to Greek text with Deepgram"),
        ("preprocess_transcripts.py", "Clean and format transcripts"),
        ("create_embeddings.py", "Create embeddings and index transcripts"),
    ]
    
    results = []
    for script_name, description in scripts:
        success = run_script(script_name, description)
        results.append((script_name, success))
        
        if not success:
            print(f"\n⚠️  Pipeline stopped due to failure in {script_name}")
            print("Fix the error and run again to continue.")
            sys.exit(1)
    
    # Print summary
    print(f"\n{'=' * 80}")
    print("Pipeline Summary:")
    print(f"{'=' * 80}")
    for script_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {script_name}")
    
    print(f"\n🎉 All pipeline stages completed successfully!")
    print("Ready to run RAG agent: uv run python agents/web_orchestrator.py")


if __name__ == "__main__":
    main()
