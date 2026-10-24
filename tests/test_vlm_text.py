"""Post-processing of the VLM's short answers into the fixed location template (no model needed)."""
import pytest

from memora.ml.vlm import compose, noun_phrase


@pytest.mark.parametrize("raw,obj,expected", [
    ("The remote control is lying on a pink blanket.", "remote control", "a pink blanket"),
    ("couch", "keys", "the couch"),
    ("Couch", "remote control", "the couch"),
    ("It is on the bedside table", "glasses", "the bedside table"),
    ("A cat.", "remote control", "a cat"),
    ("the remote control in the red box", "remote control", None),   # names the object itself -> not useful
    ("Remote", "remote control", None),                              # seen live: "next to the Remote"
    ("I don't know", "keys", None),
    ("a very long rambling answer that goes on and on about the room", "keys", None),
])
def test_noun_phrase(raw, obj, expected):
    assert noun_phrase(raw, obj) == expected


def test_compose():
    assert compose("a pink blanket", "a cat") == "on a pink blanket, next to a cat"
    assert compose(None, "a cat") == "next to a cat"
    assert compose("the shelf", "the shelf") == "on the shelf"
    assert compose(None, None) is None
