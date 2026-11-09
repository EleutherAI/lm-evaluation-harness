import pickle

import pytest

from lm_eval.decontamination.janitor import Janitor


@pytest.mark.parametrize(
    "contaminants",
    [(), ("VioletMarker",), ("ALPHA, beta", "beta gamma")],
)
def test_contamination_ngrams_round_trip(tmp_path, contaminants):
    original = Janitor(ngram_n=1)
    for text in contaminants:
        original.register_contaminant_python(text)
    filename = str(tmp_path / "ngrams.pkl")
    original.save_contamination_ngrams(filename)

    restored = Janitor(ngram_n=1)
    restored.load_contamination_ngrams(filename)
    assert restored.dirt_ngrams == original.dirt_ngrams
    restored.register_contaminant_python("ExtraMarker")
    assert restored.dirt_ngrams == original.dirt_ngrams | {"extramarker"}


def test_saved_contamination_preserves_cleaning(tmp_path):
    original = Janitor(ngram_n=1, window_to_remove=0, minimum_slice_length=0)
    original.register_contaminant_python("VioletMarker")
    text = "clean VioletMarker text"
    expected = original.clean_python(text)
    assert expected == ["clean ", " text"]
    filename = str(tmp_path / "ngrams.pkl")
    original.save_contamination_ngrams(filename)

    restored = Janitor(ngram_n=1, window_to_remove=0, minimum_slice_length=0)
    restored.load_contamination_ngrams(filename)
    assert restored.clean_python(text) == expected


def test_saved_filename_is_not_contamination(tmp_path):
    original = Janitor(ngram_n=1, window_to_remove=0, minimum_slice_length=0)
    original.register_contaminant_python("VioletMarker")
    filename = str(tmp_path / "innocent.pkl")
    original.save_contamination_ngrams(filename)

    restored = Janitor(ngram_n=1, window_to_remove=0, minimum_slice_length=0)
    restored.load_contamination_ngrams(filename)
    text = "prefix innocent suffix"
    assert original.clean_python(text) == [text]
    assert restored.clean_python(text) == [text]


def test_existing_contamination_set_file_still_loads(tmp_path):
    filename = tmp_path / "existing.pkl"
    with filename.open("wb") as handle:
        pickle.dump({"oldmarker"}, handle)
    restored = Janitor(ngram_n=1, window_to_remove=0, minimum_slice_length=0)
    restored.load_contamination_ngrams(str(filename))
    restored.register_contaminant_python("NewMarker")
    assert restored.dirt_ngrams == {"oldmarker", "newmarker"}
    assert restored.clean_python("clean OldMarker text") == ["clean ", " text"]
