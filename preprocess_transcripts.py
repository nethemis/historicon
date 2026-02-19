"""
Preprocess transcripts before creating embeddings.

This script:
- Combines consecutive lines from the same speaker
- Uses earliest start and latest end timestamps
- Removes empty lines
- Removes the FULL TRANSCRIPT section and all headers
- Creates processed files in a separate output directory
"""

import argparse
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List


@dataclass
class TimestampedEntry:
    """Represents a single timestamped transcript entry"""
    start_time: str
    end_time: str
    speaker: str
    text: str
    
    def __str__(self):
        return f"[{self.start_time} - {self.end_time}] {self.speaker}: {self.text}"


def parse_timestamped_entry(lines: List[str]) -> TimestampedEntry:
    """
    Parse a timestamped entry from lines.
    
    Args:
        lines: List of lines, first line should be timestamp + speaker,
               remaining lines are text
    
    Returns:
        TimestampedEntry object
    """
    # Parse first line: [HH:MM:SS.mmm - HH:MM:SS.mmm] Speaker N:
    first_line = lines[0]
    timestamp_pattern = r'\[(\d{2}:\d{2}:\d{2}\.\d{3}) - (\d{2}:\d{2}:\d{2}\.\d{3})\] (Speaker \d+):'
    match = re.match(timestamp_pattern, first_line)
    
    if not match:
        raise ValueError(f"Could not parse timestamp from: {first_line}")
    
    start_time = match.group(1)
    end_time = match.group(2)
    speaker = match.group(3)
    
    # Combine remaining lines as text
    text = ' '.join(line.strip() for line in lines[1:] if line.strip())
    
    return TimestampedEntry(start_time, end_time, speaker, text)


def combine_consecutive_speakers(entries: List[TimestampedEntry]) -> List[TimestampedEntry]:
    """
    Combine consecutive entries from the same speaker.
    
    Args:
        entries: List of TimestampedEntry objects
    
    Returns:
        List of combined TimestampedEntry objects
    """
    if not entries:
        return []
    
    combined = []
    current_group = [entries[0]]
    
    for entry in entries[1:]:
        if entry.speaker == current_group[0].speaker:
            # Same speaker, add to current group
            current_group.append(entry)
        else:
            # Different speaker, finalize current group and start new one
            combined.append(_merge_entries(current_group))
            current_group = [entry]
    
    # Don't forget the last group
    combined.append(_merge_entries(current_group))
    
    return combined


def _merge_entries(entries: List[TimestampedEntry]) -> TimestampedEntry:
    """Merge multiple entries into one"""
    if len(entries) == 1:
        return entries[0]
    
    start_time = entries[0].start_time
    end_time = entries[-1].end_time
    speaker = entries[0].speaker
    text = '  '.join(entry.text for entry in entries)
    
    return TimestampedEntry(start_time, end_time, speaker, text)


def preprocess_transcript(content: str) -> str:
    """
    Preprocess a transcript file.
    
    Args:
        content: Raw transcript content
    
    Returns:
        Preprocessed transcript content
    """
    lines = content.split('\n')
    
    # Find the TIMESTAMPED TRANSCRIPT WITH SPEAKERS section
    timestamped_start = -1
    for i, line in enumerate(lines):
        if 'TIMESTAMPED TRANSCRIPT WITH SPEAKERS' in line:
            timestamped_start = i
            break
    
    if timestamped_start == -1:
        raise ValueError("Could not find 'TIMESTAMPED TRANSCRIPT WITH SPEAKERS' section")
    
    # Skip the header and separator lines
    # Look for the first line that starts with [
    content_start = timestamped_start + 1
    while content_start < len(lines) and not lines[content_start].strip().startswith('['):
        content_start += 1
    
    # Parse timestamped entries
    entries = []
    current_entry_lines = []
    
    for line in lines[content_start:]:
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            continue
        
        # Check if this is a new timestamp line
        if re.match(r'^\[\d{2}:\d{2}:\d{2}\.\d{3}', stripped):
            # Save previous entry if exists
            if current_entry_lines:
                try:
                    entry = parse_timestamped_entry(current_entry_lines)
                    entries.append(entry)
                except ValueError as e:
                    print(f"Warning: {e}")
                current_entry_lines = []
            
            current_entry_lines.append(line)
        else:
            # This is a continuation of the current entry
            if current_entry_lines:
                current_entry_lines.append(line)
    
    # Don't forget the last entry
    if current_entry_lines:
        try:
            entry = parse_timestamped_entry(current_entry_lines)
            entries.append(entry)
        except ValueError as e:
            print(f"Warning: {e}")
    
    # Combine consecutive speakers
    combined_entries = combine_consecutive_speakers(entries)
    
    # Format output
    output_lines = [str(entry) for entry in combined_entries]
    return '\n'.join(output_lines)


def process_file(input_path: Path, output_dir: Path) -> bool:
    """
    Process a single transcript file.
    
    Args:
        input_path: Path to input file
        output_dir: Directory for output files
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read input file
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Preprocess
        processed_content = preprocess_transcript(content)
        
        # Write output
        output_path = output_dir / input_path.name
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(processed_content)
        
        print(f"✅ Processed: {input_path.name}")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {input_path.name}: {e}")
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Preprocess transcripts before creating embeddings"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Process only one file for testing'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('transcripts'),
        help='Input directory containing transcripts (default: transcripts)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('transcripts_processed'),
        help='Output directory for processed transcripts (default: transcripts_processed)'
    )
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    # Validate input directory
    if not input_dir.exists():
        print(f"❌ Input directory does not exist: {input_dir}")
        return 1
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Get all .txt files
    transcript_files = sorted(input_dir.glob('*.txt'))
    
    if not transcript_files:
        print(f"❌ No .txt files found in {input_dir}")
        return 1
    
    print(f"📄 Found {len(transcript_files)} transcript files")
    
    # Process files
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - Processing only first file\n")
        files_to_process = [transcript_files[0]]
    else:
        files_to_process = transcript_files
    
    success_count = 0
    for file_path in files_to_process:
        if process_file(file_path, output_dir):
            success_count += 1
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   Processed: {success_count}/{len(files_to_process)} files")
    
    if args.dry_run:
        print(f"\n💡 Dry run complete. Review the output and run without --dry-run to process all files.")
    
    return 0 if success_count == len(files_to_process) else 1


if __name__ == '__main__':
    exit(main())
