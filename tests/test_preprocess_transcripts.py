"""Tests for preprocess_transcripts.py"""

import shutil
import tempfile
from pathlib import Path

import pytest

from preprocess_transcripts import (
    TimestampedEntry,
    combine_consecutive_speakers,
    parse_timestamped_entry,
    preprocess_transcript,
)


def test_parse_timestamped_entry_simple():
    """Test parsing a simple timestamped entry"""
    lines = ["[00:01:25.780 - 00:01:27.380] Speaker 0:", "διακοπήι, εμπού,"]
    entry = parse_timestamped_entry(lines)

    assert entry.start_time == "00:01:25.780"
    assert entry.end_time == "00:01:27.380"
    assert entry.speaker == "Speaker 0"
    assert entry.text == "διακοπήι, εμπού,"


def test_parse_timestamped_entry_multiline():
    """Test parsing an entry with multiple text lines"""
    lines = ["[00:01:25.780 - 00:01:27.380] Speaker 0:", "διακοπήι, εμπού,", "επειδή"]
    entry = parse_timestamped_entry(lines)

    assert entry.text == "διακοπήι, εμπού, επειδή"


def test_combine_consecutive_speakers_same_speaker():
    """Test combining consecutive entries from same speaker"""
    entries = [
        TimestampedEntry(
            "00:01:25.780", "00:01:27.380", "Speaker 0", "διακοπήι, εμπού,"
        ),
        TimestampedEntry("00:01:27.380", "00:01:27.780", "Speaker 0", "επειδή"),
    ]

    combined = combine_consecutive_speakers(entries)

    assert len(combined) == 1
    assert combined[0].start_time == "00:01:25.780"
    assert combined[0].end_time == "00:01:27.780"
    assert combined[0].speaker == "Speaker 0"
    assert combined[0].text == "διακοπήι, εμπού,  επειδή"


def test_combine_consecutive_speakers_different_speakers():
    """Test that different speakers are not combined"""
    entries = [
        TimestampedEntry(
            "00:01:25.780", "00:01:27.380", "Speaker 0", "διακοπήι, εμπού,"
        ),
        TimestampedEntry(
            "00:01:28.740", "00:01:30.820", "Speaker 1", "μπορεί να τον πλακώσουν"
        ),
    ]

    combined = combine_consecutive_speakers(entries)

    assert len(combined) == 2
    assert combined[0].speaker == "Speaker 0"
    assert combined[1].speaker == "Speaker 1"


def test_combine_consecutive_speakers_multiple_same_then_different():
    """Test combining multiple same speakers followed by different speaker"""
    entries = [
        TimestampedEntry(
            "00:01:25.780", "00:01:27.380", "Speaker 0", "διακοπήι, εμπού,"
        ),
        TimestampedEntry("00:01:27.380", "00:01:27.780", "Speaker 0", "επειδή"),
        TimestampedEntry(
            "00:01:28.740", "00:01:30.820", "Speaker 1", "μπορεί να τον πλακώσουν"
        ),
        TimestampedEntry("00:01:30.820", "00:01:32.180", "Speaker 1", "Το ότι έγιναν"),
        TimestampedEntry(
            "00:01:32.500", "00:01:35.300", "Speaker 1", "υποκαταστάτων του Παυλίδη."
        ),
    ]

    combined = combine_consecutive_speakers(entries)

    assert len(combined) == 2

    # First speaker combined
    assert combined[0].start_time == "00:01:25.780"
    assert combined[0].end_time == "00:01:27.780"
    assert combined[0].speaker == "Speaker 0"
    assert "διακοπήι" in combined[0].text
    assert "επειδή" in combined[0].text

    # Second speaker combined
    assert combined[1].start_time == "00:01:28.740"
    assert combined[1].end_time == "00:01:35.300"
    assert combined[1].speaker == "Speaker 1"
    assert "μπορεί" in combined[1].text
    assert "Το ότι έγιναν" in combined[1].text
    assert "Παυλίδη" in combined[1].text


def test_preprocess_transcript_removes_full_transcript_section():
    """Test that FULL TRANSCRIPT section is completely removed"""
    sample_content = """================================================================================
FULL TRANSCRIPT
================================================================================
Αυτό το podcast είναι προσφορά όλων όσων μας στηρίζουν στο Patreon.

================================================================================
TIMESTAMPED TRANSCRIPT WITH SPEAKERS
================================================================================

[00:00:04.560 - 00:00:06.160] Speaker 0:
Αυτό

[00:00:09.280 - 00:00:15.185] Speaker 0:
το podcast είναι προσφορά"""

    result = preprocess_transcript(sample_content)

    assert "FULL TRANSCRIPT" not in result
    assert "TIMESTAMPED TRANSCRIPT WITH SPEAKERS" not in result
    assert "[00:00:04.560" in result


def test_preprocess_transcript_removes_empty_lines():
    """Test that empty lines are removed"""
    sample_content = """================================================================================
TIMESTAMPED TRANSCRIPT WITH SPEAKERS
================================================================================

[00:00:04.560 - 00:00:06.160] Speaker 0:
Αυτό

[00:00:09.280 - 00:00:15.185] Speaker 0:
το podcast"""

    result = preprocess_transcript(sample_content)
    lines = [line for line in result.split("\n") if line]

    # Should have no empty lines
    assert result.count("\n\n") == 0


def test_preprocess_transcript_full_example():
    """Test complete preprocessing with sample data"""
    sample_content = """================================================================================
FULL TRANSCRIPT
================================================================================
Αυτό το podcast είναι προσφορά όλων όσων μας στηρίζουν στο Patreon.

================================================================================
TIMESTAMPED TRANSCRIPT WITH SPEAKERS
================================================================================

[00:01:25.780 - 00:01:27.380] Speaker 0:
διακοπήι, εμπού,

[00:01:27.380 - 00:01:27.780] Speaker 0:
επειδή

[00:01:28.740 - 00:01:30.820] Speaker 1:
μπορεί να τον πλακώσουν ή παπαράτση.

[00:01:30.820 - 00:01:32.180] Speaker 1:
Το ότι έγιναν

[00:01:32.500 - 00:01:35.300] Speaker 1:
υποκαταστάτων του Παυλίδη.
"""

    result = preprocess_transcript(sample_content)

    # Check that headers are removed
    assert "FULL TRANSCRIPT" not in result
    assert "TIMESTAMPED TRANSCRIPT WITH SPEAKERS" not in result
    assert "================" not in result

    # Check that speakers are combined
    lines = [l for l in result.split("\n") if l.strip()]
    assert len(lines) == 2  # Should have 2 combined entries

    # Check first entry (Speaker 0 renamed to Κωνσταντίνος Ψυλλίδης)
    assert "[00:01:25.780 - 00:01:27.780] Κωνσταντίνος Ψυλλίδης:" in lines[0]
    assert "διακοπήι" in lines[0]
    assert "επειδή" in lines[0]

    # Check second entry (Speaker 1 renamed to Παύλος Παυλίδης)
    assert "[00:01:28.740 - 00:01:35.300] Παύλος Παυλίδης:" in lines[1]
    assert "μπορεί" in lines[1]
    assert "Παυλίδη" in lines[1]
