#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <deque>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

// The C++ janitor must agree with the pure-Python implementation in
// lm_eval/decontamination/janitor.py, which is the reference. Python strings
// are sequences of Unicode code points, so everything below iterates by
// code point rather than by UTF-8 byte: counting grams in bytes truncates
// multibyte characters mid-sequence (producing invalid UTF-8 that raises
// UnicodeDecodeError when pybind11 converts results back, see #1452), and
// Janitor._split_chunks slices the string by character, so the indices
// returned here must be code-point indices too.

// Length in bytes of the UTF-8 sequence starting with `lead`. Invalid lead
// bytes count as one byte, so malformed input can never split a sequence.
size_t utf8_seq_len(unsigned char lead) {
  if (lead < 0x80) return 1;
  if ((lead & 0xE0) == 0xC0) return 2;
  if ((lead & 0xF0) == 0xE0) return 3;
  if ((lead & 0xF8) == 0xF0) return 4;
  return 1;
}

// Decode the code point starting at byte `pos`. `len` is utf8_seq_len of
// its lead byte. Input coming from a Python str is valid UTF-8; anything
// malformed decodes to its first byte.
uint32_t utf8_decode(const std::string &s, size_t pos, size_t len) {
  unsigned char lead = static_cast<unsigned char>(s[pos]);
  if (len == 1) return lead;
  uint32_t cp = lead & (0x7Fu >> len);
  for (size_t k = 1; k < len && pos + k < s.size(); k++) {
    cp = (cp << 6) | (static_cast<unsigned char>(s[pos + k]) & 0x3F);
  }
  return cp;
}

// Mirrors the whitespace Python's str.split() and the regex \s class split
// on: string.whitespace (including \x1c-\x1f) plus Unicode spaces.
bool is_whitespace(uint32_t cp) {
  if (cp == 0x20 || (0x09 <= cp && cp <= 0x0D) || (0x1C <= cp && cp <= 0x1F)) {
    return true;
  }
  switch (cp) {
    case 0x85:
    case 0xA0:
    case 0x1680:
    case 0x2028:
    case 0x2029:
    case 0x202F:
    case 0x205F:
    case 0x3000:
      return true;
    default:
      return 0x2000 <= cp && cp <= 0x200A;
  }
}

// Python's translation table in Janitor.normalize_string only maps ASCII
// letters, so only ASCII bytes are lowercased here; multibyte characters
// pass through unchanged, as they do in Python.
char ascii_tolower(char ch) {
  if ('A' <= ch && ch <= 'Z') return static_cast<char>(ch - 'A' + 'a');
  return ch;
}

std::string join_grams(const std::deque<std::string> &grams) {
  std::string out;
  for (size_t i = 0; i < grams.size(); i++) {
    if (i) out += ' ';
    out += grams[i];
  }
  return out;
}

}  // namespace

// Takes a string and makes ngrams of length N, splitting grams on whitespace
// and ignoring ignored characters. Returns a LARGE array of ngrams. Matches
// word_ngrams(normalize_string(...)) in the Python implementation: grams are
// whole whitespace-delimited tokens with ignored characters removed and
// ASCII letters lowercased. (The previous version also truncated grams at
// 10 *bytes*, splitting words mid-character; Python has no such cap.)
std::vector<std::string> clean_ngram(std::string const &input,
                                     std::string const &ignore,
                                     size_t ngram_n) {
  std::vector<std::string> ngram_list;
  if (ngram_n == 0) return ngram_list;

  std::deque<std::string> window;  // completed grams in the current ngram
  std::string current_gram;
  bool started_gram = false;

  // A whitespace boundary (or end of input) completes the current gram.
  auto end_gram = [&]() {
    if (!started_gram) return;
    window.push_back(current_gram);
    current_gram.clear();
    started_gram = false;
    if (window.size() == ngram_n) {
      ngram_list.push_back(join_grams(window));
      window.pop_front();
    }
  };

  for (size_t pos = 0; pos < input.size();) {
    unsigned char lead = static_cast<unsigned char>(input[pos]);
    size_t len = utf8_seq_len(lead);
    if (pos + len > input.size()) len = 1;  // truncated final sequence
    uint32_t cp = utf8_decode(input, pos, len);

    if (is_whitespace(cp)) {
      end_gram();
    } else if (len == 1 && ignore.find(static_cast<char>(lead)) != std::string::npos) {
      // Ignored (deleted) character: contributes nothing to the gram, as
      // with Python's str.translate deletion.
    } else {
      for (size_t k = 0; k < len; k++) {
        current_gram += ascii_tolower(input[pos + k]);
      }
      started_gram = true;
    }
    pos += len;
  }
  end_gram();  // flush the final gram; Python yields trailing ngrams too

  return ngram_list;
}

// Takes a string and makes ngrams of length N, splitting grams on whitespace
// and ignoring ignored characters. Returns a LARGE array of tuples of
// (ngram, start_idx, end_idx), where the indices are code-point indices of
// the first and last characters of the raw token span, matching
// word_ngrams_indices / split_indices in the Python implementation.
std::vector<std::tuple<std::string, size_t, size_t>>
clean_ngram_with_indices(std::string const &input, std::string const &ignore,
                         size_t ngram_n) {
  std::vector<std::tuple<std::string, size_t, size_t>> ngram_list;
  if (ngram_n == 0) return ngram_list;

  struct GramSpan {
    std::string text;
    size_t start;
    size_t end;
  };
  std::deque<GramSpan> window;
  std::string current_gram;
  bool started_gram = false;
  bool in_token = false;  // inside a raw whitespace-delimited token
  size_t token_start = 0;
  size_t token_end = 0;

  auto end_gram = [&]() {
    if (!started_gram) return;
    window.push_back(GramSpan{current_gram, token_start, token_end});
    current_gram.clear();
    started_gram = false;
    if (window.size() == ngram_n) {
      std::string joined;
      for (size_t i = 0; i < window.size(); i++) {
        if (i) joined += ' ';
        joined += window[i].text;
      }
      ngram_list.push_back(
          std::make_tuple(joined, window.front().start, window.back().end));
      window.pop_front();
    }
  };

  size_t cp_index = 0;
  for (size_t pos = 0; pos < input.size();) {
    unsigned char lead = static_cast<unsigned char>(input[pos]);
    size_t len = utf8_seq_len(lead);
    if (pos + len > input.size()) len = 1;  // truncated final sequence
    uint32_t cp = utf8_decode(input, pos, len);

    if (is_whitespace(cp)) {
      end_gram();
      in_token = false;
    } else {
      if (!in_token) {
        token_start = cp_index;
        in_token = true;
      }
      token_end = cp_index;
      if (!(len == 1 &&
            ignore.find(static_cast<char>(lead)) != std::string::npos)) {
        for (size_t k = 0; k < len; k++) {
          current_gram += ascii_tolower(input[pos + k]);
        }
        started_gram = true;
      }
    }
    pos += len;
    cp_index++;
  }
  end_gram();  // flush the final gram; Python yields trailing ngrams too

  return ngram_list;
}

PYBIND11_MODULE(janitor_util, m) {
  m.doc() = "Fast ngram helpers for the janitor decontamination filter";
  m.def("clean_ngram", &clean_ngram,
        "Create ngrams of words, ignoring some characters");
  m.def("clean_ngram_with_indices", &clean_ngram_with_indices,
        "Create ngrams of words with indices, ignoring some characters");
}

// Example compile
// c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes)
// janitor_util.cpp -o janitor_util$(python3-config --extension-suffix) If
// python and gcc aren't linked, append to the above:    -undefined
// dynamic_lookup
