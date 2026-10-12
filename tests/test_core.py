
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1]
    )
)

from qc_engine import (
    normalize,
    match_required,
    read_json,
    probe_video,
    sample_frames,
    visual_scan
)


def test_normalize():
    assert (
        normalize("Mild 3-in-1")
        == "mild3in1"
    )


def test_missing_not_fail():
    assert match_required(
        "กาแฟ Mild สามอินวัน",
        ["Mild"]
    ) == []

    assert match_required(
        "กาแฟอื่น",
        ["Mild"]
    ) == ["Mild"]


def test_json():
    result = read_json(
        '```json\n'
        '{"verdict":"PASS"}'
        '\n```'
    )

    assert result["verdict"] == "PASS"
