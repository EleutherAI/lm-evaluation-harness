import os
from collections import Counter
import shutil
import glob

from lm_eval.decontamination.janitor import Janitor, word_ngrams
from scripts.clean_training_data.generate_13_grams import do_ngrams_in_buckets
from lm_eval.decontamination.archiver import Archive, TextReader

import logging

logger = logging.getLogger(__name__)


def test_generate_13_grams_1(caplog):
    data = """A goose (plural geese) is a bird of any of several waterfowl species in the family Anatidae.
    This group comprises the genera Anser (the grey geese and white geese) and Branta (the black geese).
    Some other birds, mostly related to the shelducks, have "goose" as part of their names.
    More distantly related members of the family Anatidae are swans, most of which are larger
    than true geese, and ducks, which are smaller. The term "goose" may refer to either a male
    or female bird, but when paired with "gander", refers specifically to a female one (the latter referring
    to a male). Young birds before fledging are called goslings. The collective noun for a group of
    geese on the ground is a gaggle; when in flight, they are called a skein, a team, or a wedge; when
    flying close together, they are called a plump."""

    data = data + data

    # Simple Generation
    print("simple generation")
    n = 13
    janitor = Janitor()
    ngrams = word_ngrams(janitor.normalize_string(data), n)
    comparison = list(ngrams)
    comparison_counter = Counter(comparison)
    print(len(comparison))
    # print(comparison)

    # Generating into buckets
    print("bucket generation")
    test_working_directory = "test_generate_13_grams"
    try:
        shutil.rmtree(test_working_directory)
    except FileNotFoundError:
        pass
    os.makedirs(test_working_directory)

    assert not os.path.exists("pile")
    os.makedirs("pile")
    archive = Archive(os.path.join("pile", "test.jsonl.zst"))
    archive.add_data(data)
    archive.commit()

    bucket_count = 4
    do_ngrams_in_buckets(n, test_working_directory, bucket_count)

    # Rebuild from buckets
    print("rebuild")
    rebuilt_ngrams = []
    bucket_file_paths = glob.glob(
        os.path.join(test_working_directory, "output", f"*.bkt.txt")
    )
    for bucket_file_path in bucket_file_paths:
        reader = TextReader(bucket_file_path)

        for line in reader.read():
            [ngram, document_id] = line.rsplit(" ", 1)
            rebuilt_ngrams.append(ngram)

    # Compare
    print("compare")
    result_counter = Counter(rebuilt_ngrams)
    # print(len(result_counter))
    # print(len(comparison_counter))
    assert len(result_counter) == len(comparison_counter)
    # print(result_counter)
    # print(comparison_counter)
    assert comparison_counter == result_counter
