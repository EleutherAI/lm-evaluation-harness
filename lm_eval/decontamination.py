import mmap
import string
import time
import random
import pickle
import glob
import os
import collections
import re

import tqdm

try:
    import janitor_util
    JANITOR_CPP = True
except Exception as e:
    print("WARNING: C++ module could not be loaded. Janitor running in python mode")
    JANITOR_CPP = False

# Was used for testing the evaluator decoupled from the full logic below
def get_train_overlap_stub(docs, ngrams_path, ngrams_n_size):
    simulated_overlap = 0.1
    contaminated = int(len(docs) * simulated_overlap)
    return random.sample(range(len(docs)), contaminated)


# Returns a dictionary containing all overlapping documents in each
# task based on any 13gram being found in the training set.
# ngrams_path is the parent directory containing the "ngrams_{x}.bkt.txt.sorted.zst"
# files built by the other scripts "generate_13_grams.py" and "sort_13_gram_buckets.py.
# ngrams_n_size is expected to be 13 but we made it a parameter for generality.
# The task set is only included for caching purposes.

# Algorithm:
# 1. Build lookups for each dataset {ngram: list(document_ids)}
# 2. Merge into an overall lookup {ngram: [(task_name, task_set, doc_ids),]}
# 3. Full scan the 13-grams from the training set against the merged lookup,
#    saving matches in the "duplicates" dictionary {(task_name, task_set): set(doc_ids)}
# 4. Strip the task_set from the dictionary keys and return
#
# We cache the task+set lookups as well as the overlaps.
#
# Currently calculating some per file ngram stats for interest, might remove before merging into main
def get_train_overlap(docs_by_task_set, ngrams_path, ngrams_n_size, limit):
    # return get_train_overlap_stub(docs, ngrams_path, ngrams_n_size)

    janitor = Janitor()

    # Build lookup for each dataset first in case we use different task combinations later
    print("Building Lookups...")
    start = time.perf_counter()

    def get_overlaps_dump_path(task_name, task_set, ngrams_n_size, limit):
        return f"data/{task_name}/{task_set}_{ngrams_n_size}grams_limit{limit}.overlaps"

    lookups = {}
    duplicates = {} # (task_name, task_set): set(doc_ids)}
    sets_to_decontaminate = len(docs_by_task_set.keys())

    for (task_name, task_set), docs in docs_by_task_set.items():
        if not os.path.exists(f"data/{task_name}"):
            os.mkdir(f"data/{task_name}")
        # Check if we've decontaminated this set before
        overlaps_dump_path = get_overlaps_dump_path(task_name, task_set, ngrams_n_size, limit)
        if os.path.exists(overlaps_dump_path):
            duplicates[(task_name, task_set)] = pickle.load(open(overlaps_dump_path, "rb"))
            sets_to_decontaminate -= 1
            continue
        else:
            duplicates[(task_name, task_set)] = set() # No defaultdict, we want to dump empty sets too later

        # Build/load the task lookup {ngram: documents}.
        task_set_lookup_path = f"data/{task_name}/{task_set}_{ngrams_n_size}grams_limit{limit}.lookup"
        if os.path.exists(task_set_lookup_path):
            print(f"{task_set_lookup_path} available, loading...")        
            lookups[(task_name, task_set)] = pickle.load(open(task_set_lookup_path, "rb"))
        else:
            print(f"{task_set_lookup_path} not available, building...")
            lookup = collections.defaultdict(set)

            for doc_id, document in enumerate(docs):
                ngrams = word_ngrams(janitor.normalize_string(document), ngrams_n_size)
                for ngram in ngrams:
                    lookup[ngram].add(doc_id)

            pickle.dump(lookup, open(task_set_lookup_path,"wb"))
            lookups[(task_name, task_set)] = lookup

    elapsed = time.perf_counter() - start
    print(f"Building lookups took {elapsed:0.5f} seconds.")

    matched_ngrams = []

    if sets_to_decontaminate > 0:
        print("Merging lookups...")
        start = time.perf_counter()    
        merged_lookup = collections.defaultdict(list)
        for (task_name, task_set), lookup in lookups.items():
            for ngram, doc_ids in lookup.items():
                merged_lookup[ngram].append((task_name, task_set, doc_ids))

        elapsed = time.perf_counter() - start
        print(f"Merging lookups took {elapsed:0.5f} seconds.")

        print(f"13 grams files found in {ngrams_path}:")
        files = glob.glob(os.path.join(ngrams_path, f"*.sorted.zst"))
        print(files)

        for file in files:
            start = time.perf_counter()
            print(f"Scanning {file}")
            reader = ZStdTextReader(file)
            total_ngrams = 0
            unique_ngrams = 0
            matching_unique = 0
            non_matching_unique = 0

            current_ngram = ""
            for line in reader.read_tqdm(): # Scan training set ngrams file
                total_ngrams += 1
                [ngram, document_id] = line.rsplit(" ", 1)
                if ngram != current_ngram: # Only need to match the ngram once in training set
                    unique_ngrams += 1
                    current_ngram = ngram
                    if ngram in merged_lookup:
                        matched_ngrams.append(ngram) # For logging
                        matching_unique += 1
                        for task_name, task_set, doc_ids in merged_lookup[ngram]:
                            task_doc_set = duplicates[(task_name, task_set)]
                            for doc_id in doc_ids: # Record contamination across all relevant task/set combos
                                task_doc_set.add(doc_id)
                        del merged_lookup[ngram] # No point matching again
                    else:
                        non_matching_unique += 1

            print(f"Total Ngrams: {total_ngrams}")
            print(f"Unique Ngrams: {unique_ngrams}")
            print(f"Unique Matching: {matching_unique}")
            print(f"Unique Non Matching: {non_matching_unique}")
            print("Matched ngrams:")
            for ngram in matched_ngrams:
                print(ngram)

            elapsed = time.perf_counter() - start
            print(f"Read took {elapsed:0.5f} seconds.")
            print(f"Speed: {(os.path.getsize(file)/1000000.0)/elapsed}MB/second")

        print(duplicates)

        # Dump overlaps separately    
        for (task_name, task_set), doc_ids in duplicates.items():
            overlaps_dump_path = get_overlaps_dump_path(task_name, task_set, ngrams_n_size, limit)
            pickle.dump(doc_ids, open(overlaps_dump_path,"wb"))

    # Strip task set and return
    return {task_name: doc_ids for (task_name, task_set), doc_ids in duplicates.items()}


# Implementation from nltk source
# https://www.nltk.org/_modules/nltk/util.html
def form_ngrams(sequence, n):
    history = []
    while n > 1:
        # PEP 479, prevent RuntimeError from being raised when StopIteration bubbles out of generator
        try:
            next_item = next(sequence)
        except StopIteration:
            # no more data, terminate the generator
            return
        history.append(next_item)
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]


def word_ngrams(s, n):
    """Splits a string into ngram words"""
    tokens = s.split()  # not a generator :(
    ngram_seqs = form_ngrams(iter(tokens), n)
    return (" ".join(ngram) for ngram in ngram_seqs)


# https://stackoverflow.com/questions/13734451/string-split-with-indices-in-python
def split_indices(s):
    """Splits a string on whitespaces and records the indices of each in the original string.
    @:return generator((word, (start_idx, end_idx)), ...)
    """
    return ((m.group(0), (m.start(), m.end() - 1)) for m in re.finditer(r'\S+', s))


def word_ngrams_indices(s, n):
    """Splits a string into pairs of (ngram words, their start/end indices)"""
    tokens_with_indices = split_indices(s)

    # Generator of ngrams of (word, idx_pairs)
    # (
    #   [(word, (start,end)), (word, (start, end))...],
    #   [(word, (start, end)), ...],
    #   ...
    # )
    ngram_seqs_with_indices = form_ngrams(tokens_with_indices, n)

    # Generator of pairs of word and index ngrams
    # (
    #   ([word, word, ...], [(start,end), (start,end), ...]),
    #   ...
    # )
    ngram_indices_pairs = (zip(*ngram_with_indices) for ngram_with_indices in ngram_seqs_with_indices)

    # Generator of ( (word_ngram, (start, end)), (word_ngram, start, end)), ...)
    return ((" ".join(ngram_seq), (indices[0][0], indices[-1][1])) for ngram_seq, indices in ngram_indices_pairs)


class Janitor:

    # FIXME delete_chars: Should anything else go here? Special chars?
    def __init__(
            self,
            ngram_n=13,
            window_to_remove=200,
            too_dirty_cutoff=10,
            minimum_slice_length=200,
            delete_chars=string.punctuation
    ):
        self.ngram_n = ngram_n
        self.window_to_remove = window_to_remove
        self.too_dirty_cutoff = too_dirty_cutoff
        self.minimum_slice_length = minimum_slice_length
        self.delete_chars = delete_chars

        self.dirt_ngrams = set()

        # If in python, we'll translate uppercase to lowercase and delete naughty characters.
        # This is fast by python standards
        # https://stackoverflow.com/questions/638893/what-is-the-most-efficient-way-in-python-to-convert-a-string-to-all-lowercase-st
        self.translation_table = str.maketrans(
            string.ascii_lowercase + string.ascii_uppercase,  # These characters
            string.ascii_lowercase * 2,  # Become these characters
            self.delete_chars  # These are deleted
        )

    ##############
    # I/O for saving contamination ngrams
    ##############

    def save_contamination_ngrams(self, filename):
        with open(filename, 'wb') as fp:
            pickle.dump(filename, fp)

    def load_contamination_ngrams(self, filename):
        with open(filename, 'rb') as fp:
            self.dirt_ngrams = pickle.load(fp)


    ##############
    # Call these :)
    ##############

    def register_contaminant(self, dirt_string):
        """Register a string as contamination to be removed, e.g. a test set
        This breaks the dirt_string into ngrams to store for future cleaning"""
        if JANITOR_CPP:
            return self.register_contaminant_cpp(dirt_string)
        else:
            print("WARNING: Janitor running in python mode")
            return self.register_contaminant_python(dirt_string)

    def clean(self, dirty_string):
        """Clean a string (e.g. a training set) by removing all ngrams previously
        reigstered as contaminants. Returns a list of clean chunks, or empty if
        the string was too dirty"""
        if JANITOR_CPP:
            return self.clean_cpp(dirty_string)
        else:
            print("WARNING: Janitor running in python mode")
            return self.clean_python(dirty_string)

    def _split_chunks(self, dirty_string, dirty_parts):
        clean_chunks = []
        splice_idx = 0
        end = -1
        for i, (ngram, start, end) in enumerate(dirty_parts):
            if i >= self.too_dirty_cutoff:
                return []
            start = max(0, start - self.window_to_remove)
            end = min(len(dirty_string), end + self.window_to_remove)

            if start - splice_idx > self.minimum_slice_length:
                clean_chunks.append(dirty_string[splice_idx: start])
            splice_idx = end

        if end < len(dirty_string) - self.minimum_slice_length:
            clean_chunks.append(dirty_string[end+1:])

        return clean_chunks

    ##############
    # Fast C++
    ##############

    def register_contaminant_cpp(self, dirt_string):
        self.dirt_ngrams.update(janitor_util.clean_ngram(dirt_string, self.delete_chars, self.ngram_n))

    def clean_cpp(self, dirty_string):
        contamination_indices = janitor_util.clean_ngram_with_indices(dirty_string, self.delete_chars, self.ngram_n)
        return self._split_chunks(dirty_string, contamination_indices)

    ##############
    # Slow python
    ##############

    def normalize_string(self, s):
        return s.translate(self.translation_table)

    def register_contaminant_python(self, dirt_string):
        self.dirt_ngrams.update(word_ngrams(self.normalize_string(dirt_string), self.ngram_n))

    def clean_python(self, dirty_string):
        contamination_indices = (
            (None, *idx_pair)
            for dirty_ngram, idx_pair in word_ngrams_indices(dirty_string, self.ngram_n)
            if self.normalize_string(dirty_ngram) in self.dirt_ngrams
        )
        return self._split_chunks(dirty_string, contamination_indices)



# Implementation from nltk source
# https://www.nltk.org/_modules/nltk/util.html
def form_ngrams(sequence, n):
    history = []
    while n > 1:
        # PEP 479, prevent RuntimeError from being raised when StopIteration bubbles out of generator
        try:
            next_item = next(sequence)
        except StopIteration:
            # no more data, terminate the generator
            return
        history.append(next_item)
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]


def word_ngrams(s, n):
    """Splits a string into ngram words"""
    tokens = s.split()  # not a generator :(
    ngram_seqs = form_ngrams(iter(tokens), n)
    return (" ".join(ngram) for ngram in ngram_seqs)

# Simple text reader and writer with same interface as above
class TextArchive:
    def __init__(self, file_path, mode="ab"):
        self.file_path = file_path
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)    
        self.fh = open(self.file_path, mode)      
    
    def add_data(self, data, meta={}):
        self.fh.write(data.encode('UTF-8') + b'\n')
    
    def commit(self):
        self.fh.flush()
        self.fh.close()

class TextReader:
    def __init__(self, file_path):
        self.file_path = file_path

    # Optimized mmap read with infrequent tqdm updates to maintain speed
    # Tested up to 250MB/s. 
    def read_tqdm(self, update_frequency=10000):
        current_file_position = 0
        line_counter = 0
        with open(self.file_path, 'r') as fh, \
            tqdm.tqdm(total=os.path.getsize(self.file_path), dynamic_ncols=True, 
                unit="byte", unit_scale=1) as progress:
            with mmap.mmap(fh.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_obj:
                for line in iter(mmap_obj.readline, b""):
                    line = line.decode("utf-8")
                    line_counter += 1
                    if line_counter == update_frequency:
                        new_file_pos = mmap_obj.tell() 
                        bytes_read = new_file_pos - current_file_position
                        current_file_position = new_file_pos
                        progress.update(bytes_read)
                        line_counter = 0
                    yield line[:-1]

    def read_and_tell(self):
        current_file_position = 0
        with open(self.file_path, 'r', encoding="utf8") as fh:
            with mmap.mmap(fh.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_obj:
                for line in iter(mmap_obj.readline, b""):
                    line = line.decode("utf-8")                    
                    new_file_pos = mmap_obj.tell() 
                    raw_bytes_read = new_file_pos - current_file_position
                    current_file_position = new_file_pos
                    yield line[:-1], raw_bytes_read

    def read(self):
        with open(self.file_path, 'r', encoding="utf8") as fh:
            with mmap.mmap(fh.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_obj:
                for line in iter(mmap_obj.readline, b""):
                    line = line.decode("utf-8")                    
                    yield line[:-1]

    def read_slow(self):
        with open(self.file_path, 'r', encoding="utf8") as fh:
            while True:
                line = fh.readline()
                if line == -1 or line == "":
                    break
                else:
                    yield line[:-1]

# Optimized for speed. Decompresses the archive in shell before
# using the mmap'd TextReader.
class ZStdTextReader:
    def __init__(self, file):
        self.file = file        

    def read_tqdm(self):       
        decompressed_file = self.file[:-4]
        print("Decompressing file, please wait...")
        os.system(f"zstd -d {self.file}") # linux decompress is faster        
        reader = TextReader(decompressed_file)
        yield from reader.read_tqdm()
        os.remove(decompressed_file)

