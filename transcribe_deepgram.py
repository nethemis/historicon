import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from deepgram import DeepgramClient


def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS.mmm format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def transcribe_audio(audio_file_path, output_dir="transcripts"):
    """
    Transcribe Greek audio using Deepgram's Nova-3 model.

    Args:
        audio_file_path: Path to the audio file to transcribe
        output_dir: Directory to save the transcript

    Returns:
        tuple: (status, filename, message, transcript_length)
    """
    try:
        # Initialize Deepgram client (automatically reads DEEPGRAM_API_KEY from environment)
        client = DeepgramClient()

        # Read and transcribe the audio file
        with open(audio_file_path, "rb") as audio_file:
            response = client.listen.v1.media.transcribe_file(
                request=audio_file.read(),
                model="nova-3",
                language="el",  # Greek
                smart_format=True,
                punctuate=True,
                paragraphs=True,
                utterances=True,
                diarize=True,  # Speaker diarization
            )

        # Extract the transcript
        transcript = response.results.channels[0].alternatives[0].transcript

        # Get output filename
        base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}.txt")

        # Save the transcript with timestamps and speaker separation
        with open(output_file, "w", encoding="utf-8") as f:
            # Write plain transcript at the top
            f.write("=" * 80 + "\n")
            f.write("FULL TRANSCRIPT\n")
            f.write("=" * 80 + "\n")
            f.write(transcript + "\n\n")

            # Write timestamped utterances with speaker labels
            f.write("=" * 80 + "\n")
            f.write("TIMESTAMPED TRANSCRIPT WITH SPEAKERS\n")
            f.write("=" * 80 + "\n\n")

            # Check if utterances are available
            if hasattr(response.results, "utterances") and response.results.utterances:
                for utterance in response.results.utterances:
                    # Format timestamp
                    start_time = format_timestamp(utterance.start)
                    end_time = format_timestamp(utterance.end)
                    speaker = (
                        f"Speaker {utterance.speaker}"
                        if hasattr(utterance, "speaker")
                        else "Unknown"
                    )

                    f.write(f"[{start_time} - {end_time}] {speaker}:\n")
                    f.write(f"{utterance.transcript}\n\n")
            else:
                # Fallback to words if utterances not available
                words = response.results.channels[0].alternatives[0].words
                if words:
                    current_speaker = None
                    current_text = []
                    current_start = None

                    for word in words:
                        speaker = (
                            f"Speaker {word.speaker}"
                            if hasattr(word, "speaker")
                            else "Unknown"
                        )

                        if speaker != current_speaker:
                            # Write previous speaker's segment
                            if current_text:
                                start_time = format_timestamp(current_start)
                                end_time = format_timestamp(
                                    words[words.index(word) - 1].end
                                )
                                f.write(
                                    f"[{start_time} - {end_time}] {current_speaker}:\n"
                                )
                                f.write(f"{' '.join(current_text)}\n\n")

                            # Start new speaker segment
                            current_speaker = speaker
                            current_text = [word.word]
                            current_start = word.start
                        else:
                            current_text.append(word.word)

                    # Write last segment
                    if current_text:
                        start_time = format_timestamp(current_start)
                        end_time = format_timestamp(words[-1].end)
                        f.write(f"[{start_time} - {end_time}] {current_speaker}:\n")
                        f.write(f"{' '.join(current_text)}\n\n")

        return ("success", base_name, None, len(transcript))

    except FileNotFoundError:
        return ("error", os.path.basename(audio_file_path), "File not found", 0)
    except Exception as e:
        return ("error", os.path.basename(audio_file_path), str(e), 0)


def batch_transcribe(input_dir="inputs", output_dir="transcripts", max_workers=5):
    """
    Transcribe all audio files in input directory using parallel processing.

    Args:
        input_dir: Directory containing audio files
        output_dir: Directory to save transcripts
        max_workers: Number of concurrent transcriptions
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Supported audio formats
    audio_extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".webm"}

    # Find all audio files
    input_path = Path(input_dir)
    audio_files = [
        f
        for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in audio_extensions
    ]

    if not audio_files:
        print(f"No audio files found in {input_dir}/")
        return

    print(f"Found {len(audio_files)} audio files in {input_dir}/\n")

    # Filter out already transcribed files
    transcribe_tasks = []
    skipped = 0

    for audio_file in audio_files:
        base_name = audio_file.stem
        output_file = Path(output_dir) / f"{base_name}.txt"

        if output_file.exists():
            print(f"⏭️  Skipping: {audio_file.name} (transcript exists)")
            skipped += 1
        else:
            transcribe_tasks.append(str(audio_file))

    if not transcribe_tasks:
        print(f"\n✓ All files already transcribed!")
        return

    print(
        f"\nStarting transcription of {len(transcribe_tasks)} files ({max_workers} concurrent)...\n"
    )

    # Transcribe files in parallel
    completed = 0
    errors = 0
    total_chars = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all transcription tasks
        future_to_file = {
            executor.submit(transcribe_audio, file_path, output_dir): file_path
            for file_path in transcribe_tasks
        }

        # Process completed transcriptions as they finish
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            status, filename, message, char_count = future.result()

            if status == "success":
                print(f"✅ Transcribed: {filename} ({char_count:,} characters)")
                completed += 1
                total_chars += char_count
            elif status == "error":
                print(f"❌ Error transcribing {filename}: {message}")
                errors += 1

    print("\n" + "=" * 70)
    print(f"Transcription complete!")
    print(f"Completed: {completed} files")
    print(f"Skipped: {skipped} files (already transcribed)")
    if errors > 0:
        print(f"Errors: {errors} files")
    print(f"Total characters transcribed: {total_chars:,}")
    print(f"All transcripts saved in: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    batch_transcribe()
