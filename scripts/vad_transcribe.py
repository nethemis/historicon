import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import mlx_whisper
import torch
import torchaudio
from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

SAMPLE_RATE = 16000
CHUNK_SECONDS = 20 * 60  # 20 minutes per chunk

import logging

log = logging.getLogger("vad_transcribe")


def segments_to_srt(segments) -> str:
    """Convert mlx_whisper segments to SRT format."""
    srt_lines = []
    for i, segment in enumerate(segments, 1):
        start_time = ms_to_srt(int(segment["start"] * 1000))
        end_time = ms_to_srt(int(segment["end"] * 1000))
        text = segment["text"].strip()

        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(text)
        srt_lines.append("")  # Empty line between segments

    return "\n".join(srt_lines).rstrip()


def run_mlx_whisper(audio_path: str, output_dir: str) -> Path:
    # Get absolute path to the local model
    model_path = (
        Path(__file__).parent.parent
        / "greek_whisper"
        / "mlx-examples"
        / "whisper"
        / "greek-whisper-mlx"
    )

    log.info(f"Running mlx_whisper with model: {model_path}")

    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=str(model_path),
        language="el",
        verbose=True,
        word_timestamps=True,
        condition_on_previous_text=False,
        no_speech_threshold=0.7,
        compression_ratio_threshold=1.6,
        temperature=(0.0, 0.2, 0.4),
        initial_prompt=(
            "Podcast στα κυπριακά ελληνικά. Δύο άντρες συζητούν για πολιτικά και επικαιρότητα της Κύπρου. "
            "Μιλούν με κυπριακή διάλεκτο. "
            "Χαρακτηριστικές λέξεις: τζιαί, εν, εν έτσι, εννά, κάμνω, τζείνος, τούτος, "
            "κόπελλος, κόπελλα, παιδκιά, Κύπρος, Κυπριακό, Σάντος, George Santos, "
            "ιστορικό, podcast, Κωνσταντίνος, Παύλος."
        ),
    )

    # Convert result to SRT format
    srt_text = segments_to_srt(result["segments"])

    # Write SRT file
    output_path = Path(output_dir) / (Path(audio_path).stem + ".srt")
    output_path.write_text(srt_text, encoding="utf-8")

    return output_path


def build_time_map(timestamps: list[dict], chunk_offset_seconds: float) -> list[tuple]:
    """
    Build a mapping from compressed-audio seconds → original-file seconds.

    Each entry is (compressed_start, compressed_end, original_start)
    meaning: compressed_audio[compressed_start:compressed_end]
             corresponds to original_audio[original_start:original_start+(compressed_end-compressed_start)]
    """
    time_map = []
    compressed_cursor = 0.0
    for ts in timestamps:
        orig_start = chunk_offset_seconds + ts["start"] / SAMPLE_RATE
        duration = (ts["end"] - ts["start"]) / SAMPLE_RATE
        time_map.append((compressed_cursor, compressed_cursor + duration, orig_start))
        compressed_cursor += duration
    return time_map


def compressed_to_original(t: float, time_map: list[tuple]) -> float:
    """Map a compressed-audio timestamp (seconds) to original-file time (seconds)."""
    for comp_start, comp_end, orig_start in time_map:
        if comp_start <= t <= comp_end:
            return orig_start + (t - comp_start)
    # If beyond last segment, clamp to end of last segment
    if time_map:
        comp_start, comp_end, orig_start = time_map[-1]
        return orig_start + (comp_end - comp_start)
    return t


def ms_to_srt(ms: int) -> str:
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def srt_to_ms(h, m, s, ms) -> int:
    return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)


def remap_srt_timestamps(srt_text: str, time_map: list[tuple]) -> str:
    """Remap all SRT timestamps using the VAD time_map."""
    pattern = r"(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})"

    def remap(match):
        start_ms = srt_to_ms(*match.group(1, 2, 3, 4))
        end_ms = srt_to_ms(*match.group(5, 6, 7, 8))
        new_start = compressed_to_original(start_ms / 1000, time_map)
        new_end = compressed_to_original(end_ms / 1000, time_map)
        return (
            f"{ms_to_srt(int(new_start * 1000))} --> {ms_to_srt(int(new_end * 1000))}"
        )

    return re.sub(pattern, remap, srt_text)


def renumber_srt(srt_text: str, start_index: int) -> tuple[str, int]:
    blocks = re.split(r"\n\n+", srt_text.strip())
    out = []
    i = start_index
    for block in blocks:
        if block.strip():
            lines = block.strip().split("\n")
            lines[0] = str(i)
            out.append("\n".join(lines))
            i += 1
    return "\n\n".join(out), i


def vad_filter_and_transcribe(
    input_path: str, output_dir: str, max_seconds: int = None
):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading VAD model...")
    model = load_silero_vad()
    log.info("Reading audio...")
    wav = read_audio(str(input_path))

    if max_seconds:
        wav = wav[: max_seconds * SAMPLE_RATE]

    total_samples = wav.shape[0]
    chunk_samples = CHUNK_SECONDS * SAMPLE_RATE
    n_chunks = (total_samples + chunk_samples - 1) // chunk_samples

    log.info(
        f"Audio: {total_samples/SAMPLE_RATE/60:.1f} min total, {n_chunks} chunk(s)"
    )

    combined_srt = ""
    srt_index = 1
    tmp_files = []

    try:
        for i in range(n_chunks):
            chunk_start = i * chunk_samples
            chunk_end = min((i + 1) * chunk_samples, total_samples)
            chunk_offset_seconds = chunk_start / SAMPLE_RATE

            log.info(f"Chunk {i+1}/{n_chunks}: VAD filtering...")
            chunk_wav = wav[chunk_start:chunk_end]

            timestamps = get_speech_timestamps(
                chunk_wav,
                model,
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=500,
                return_seconds=False,
            )

            if not timestamps:
                log.info(f"Chunk {i+1}/{n_chunks}: no speech detected, skipping")
                continue

            # Build time map BEFORE concatenating speech
            time_map = build_time_map(timestamps, chunk_offset_seconds)

            speech_only = torch.cat(
                [chunk_wav[ts["start"] : ts["end"]] for ts in timestamps]
            )

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            tmp_files.append(tmp_path)
            torchaudio.save(tmp_path, speech_only.unsqueeze(0), SAMPLE_RATE)

            log.info(f"Chunk {i+1}/{n_chunks}: transcribing...")
            with tempfile.TemporaryDirectory() as tmp_out:
                srt_path = run_mlx_whisper(tmp_path, tmp_out)
                srt_text = srt_path.read_text(encoding="utf-8")

            # Remap compressed timestamps → original file timestamps
            srt_text = remap_srt_timestamps(srt_text, time_map)
            srt_text, srt_index = renumber_srt(srt_text, srt_index)
            combined_srt += srt_text + "\n\n"

    finally:
        for f in tmp_files:
            if os.path.exists(f):
                os.unlink(f)

    out_file = output_dir / (input_path.stem + ".srt")
    out_file.write_text(combined_srt.strip(), encoding="utf-8")
    log.info(f"Done. Written to {out_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output_dir")
    parser.add_argument("max_seconds", nargs="?", type=int, default=None)
    parser.add_argument("--logs", action="store_true")
    args = parser.parse_args()

    if args.logs:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
        )

    vad_filter_and_transcribe(args.input, args.output_dir, args.max_seconds)
