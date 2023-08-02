#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <queue>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

bool is_whitespace(char ch) noexcept {
  // " \t\n\r\x0b\x0c" (python string.whitespace)
  return ch == 32 or (9 <= ch and ch <= 13);
  //    return ch <= 32; // arguably too general, but slightly faster
}

bool is_punctuation(char c) noexcept {
  // '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'      ascii values:    33-47,  58-64,
  // 91-96,  123-126
  return (33 <= c and c <= 47) or (58 <= c and c <= 64) or
         (91 <= c and c <= 96) or (123 <= c and c <= 126);
}

// Takes a string and makes ngrams of length N, splitting grams on whitespace
// and ignoring ignored characters Returns a LARGE array of ngrams
std::vector<std::string> clean_ngram(std::string const &input,
                                     std::string const &ignore,
                                     size_t ngram_n) noexcept {

  size_t num_grams = 0;
  std::vector<std::string> ngram_list;
  std::vector<uint8_t> gram_lengths;
  std::string current_ngram;

  // Max gram length is set to 10 below.
  current_ngram.reserve(11 * ngram_n);
  gram_lengths.reserve(ngram_n);

  bool started_gram = false;
  gram_lengths.push_back(0);

  // for (size_t i=0; i<input.length(); i++) {
  //  this is slightly faster, and we don't need the index in this one
  for (auto iter = input.begin(); iter != input.end(); iter++) {

    // If whitespace, end the current ngram and start the next
    // alternatively, (perhaps marginally) faster: if (is_whitespace(ch)) { ...
    // }
    if (is_whitespace(*iter) || gram_lengths.back() > 10) {

      // Skip all whitespace
      while (++iter != input.end() && is_whitespace(*iter))
        ;
      iter--;

      if (started_gram) {
        num_grams += 1;

        // Building 1grams is a special case
        if (ngram_n == 1) {
          ngram_list.push_back(current_ngram);
          current_ngram = current_ngram.substr(gram_lengths.front());
          gram_lengths.back() = 0;

          // If there are enough grams to form an ngram, save
        } else if (num_grams >= ngram_n) {
          // Save the current ngram
          ngram_list.push_back(current_ngram);

          // Start the next ngram by dropping the first gram and its space from
          // the ngram
          current_ngram = current_ngram.substr(gram_lengths.front() + 1);
          current_ngram += ' ';

          // Drop the length of the first gram and prepare to record the length
          // of the new gram
          gram_lengths.erase(gram_lengths.begin());
          gram_lengths.push_back(0);

          // Otherwise, continute building
        } else {
          current_ngram += ' ';
          gram_lengths.push_back(0);
        }

        started_gram = false;
      }

      // Skip ignored characters
      // alternatively, (perhaps marginally) faster: if (is_punctuation(ch))
      // continue;
    } else if (ignore.find(*iter) != std::string::npos) {
      continue;
    }

    // If it is a non-ignored character, add it to the ngram and update the last
    // gram's length
    else {
      current_ngram += tolower(*iter);
      gram_lengths.back() += 1;
      started_gram = true;
    }
  }

  return ngram_list;
}

// Takes a string and makes ngrams of length N, splitting grams on whitespace
// and ignoring ignored characters Returns a LARGE array of tuples of (ngram,
// start_idx, end_idx)
std::vector<std::tuple<std::string, size_t, size_t>>
clean_ngram_with_indices(std::string const &input, std::string const &ignore,
                         size_t ngram_n) noexcept {

  size_t num_grams = 0;
  std::vector<std::tuple<std::string, size_t, size_t>> ngram_list;
  std::vector<uint8_t> gram_lengths;
  std::vector<size_t> gram_start_indices;
  std::string current_ngram;

  // Max gram length is set to 10 below.
  current_ngram.reserve(11 * ngram_n);

  bool started_gram = false;
  gram_lengths.push_back(0);
  gram_start_indices.push_back(0);

  for (size_t i = 0; i < input.length(); i++) {
    char ch = input[i];

    // If whitespace, end the current ngram and start the next
    if (is_whitespace(ch) || gram_lengths.back() > 10) {

      // Skip all whitespace
      while (++i < input.length() && is_whitespace(input[i]))
        ;
      i--;

      if (started_gram) {
        num_grams += 1;

        // Building 1grams is a special case
        if (ngram_n == 1) {
          ngram_list.push_back(
              std::make_tuple(current_ngram, gram_start_indices.front(), i));
          current_ngram = current_ngram.substr(gram_lengths.front());
          gram_lengths.back() = 0;
          gram_start_indices.back() = i + 1;

          // If there are enough grams to form an ngram, save
        } else if (num_grams >= ngram_n) {

          // Save the current ngram
          ngram_list.push_back(
              std::make_tuple(current_ngram, gram_start_indices.front(), i));

          // Start the next ngram by dropping the first gram and its space from
          // the ngram
          current_ngram = current_ngram.substr(gram_lengths.front() + 1);
          current_ngram += ' ';

          // Drop the length of the first gram and prepare to record the length
          // of the new gram
          gram_lengths.erase(gram_lengths.begin());
          gram_lengths.push_back(0);

          gram_start_indices.erase(gram_start_indices.begin());
          gram_start_indices.push_back(i + 1);

          // Otherwise, continute building
        } else {
          current_ngram += ' ';
          gram_lengths.push_back(0);
          gram_start_indices.push_back(i + 1);
        }

        started_gram = false;
      }

      // Skip ignored characters
    } else if (ignore.find(ch) != std::string::npos) {
      continue;

      // If it is a non-ignored character, add it to the ngram and update the
      // last gram's length
    } else {
      current_ngram += tolower(ch);
      gram_lengths.back() += 1;
      started_gram = true;
    }
  }

  return ngram_list;
}

PYBIND11_MODULE(janitor_util, m) {
  m.doc() = "pybind11 example plugin"; // optional module docstring
  //    m.def("add", &add, "A function which adds two numbers");  // example
  //    function
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
