import abc
import json
from lm_eval.utils import sh
from lm_eval.metrics import mean
from lm_eval.base import Task, rf
from pathlib import Path
from best_download import download_file


class Math(Task):
    """
    This dataset is based on the following paper:
    https://arxiv.org/abs/2103.03874 
    """

    DATASET_PATH = Path('data/MATH')

    def download(self):
        if not (self.DATASET_PATH / 'test').exists() or not (self.DATASET_PATH / 'done').exists():
            sh(f"mkdir -p {self.DATASET_PATH}")
            download_file("https://people.eecs.berkeley.edu/~hendrycks/MATH.tar", f"{self.DATASET_PATH}.tar", "01256fd7cd5430596fdf07e6e6a5827111b5235b7ffed679c662a12f898932da")
            sh(f"""
            tar -xf {self.DATASET_PATH}.tar -C data/ && touch {self.DATASET_PATH / 'done'}
            rm {self.DATASET_PATH}.tar
            """)

    @abc.abstractmethod
    def get_file_info(self):
        """returns directory name"""
        pass

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def _load_docs(self, path):
        for file in sorted(path.iterdir()):
            with open(file) as f:
                doc = json.load(f)
                doc["answer"] = self.remove_boxed(
                    self.last_boxed_only_string(doc["solution"]))
                yield doc

    def training_docs(self):
        return self._load_docs(self.DATASET_PATH / "train" / self.get_file_info())

    def validation_docs(self):
        return NotImplemented

    def test_docs(self):
        return self._load_docs(self.DATASET_PATH / "test" / self.get_file_info())

    def fewshot_description(self):
        return "Given a mathematics problem, determine the answer. Simplify your answer as much as possible."

    def doc_to_text(self, doc):
        return "Problem: " + doc["problem"] + "\nAnswer:"

    def doc_to_target(self, doc):
        return " " + doc["answer"]

    def construct_requests(self, doc, ctx):
        return rf.greedy_until(ctx, ["\n"])

    def process_results(self, doc, results):
        retval = 0
        indices = [pos for pos, char in enumerate(results[0]) if char == "$"]
        if len(indices) <= 1:
            answer = results[0]
        else:
            answer = results[0][indices[0]+1:indices[-1]]

        if self.is_equiv(answer, self.remove_boxed(self.last_boxed_only_string(doc["solution"]))):
            retval = 1
        return {
            "acc": retval
        }

    def aggregation(self):
        return {
            'acc': mean
        }

    def higher_is_better(self):
        return {
            'acc': True
        }

    def is_equiv(self, str1, str2, verbose=False):
        if str1 is None and str2 is None:
            print("WARNING: Both None")
            return True
        if str1 is None or str2 is None:
            return False

        try:
            ss1 = self.strip_string(str1)
            ss2 = self.strip_string(str2)
            if verbose:
                print(ss1, ss2)
            return ss1 == ss2
        except:
            return str1 == str2

    def remove_boxed(self, s):
        if "\\boxed " in s:
            left = "\\boxed "
            assert s[:len(left)] == left
            return s[len(left):]

        left = "\\boxed{"

        assert s[:len(left)] == left
        assert s[-1] == "}"

        return s[len(left):-1]

    def last_boxed_only_string(self, string):
            
        idx = string.rfind("\\boxed")
        if "\\boxed " in string:
            return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is None:
            retval = None
        else:
            retval = string[idx:right_brace_idx + 1]

        return retval

    def fix_fracs(self, string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except AssertionError:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string

    def fix_a_slash_b(self, string):
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except AssertionError:
            return string

    def remove_right_units(self, string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    def fix_sqrt(self, string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    class NotEqual:
        def __eq__(self, other):
            return False

    def strip_string(self, string):
        # linebreaks
        string = string.replace("\n", "")

        # remove inverse spaces
        string = string.replace("\\!", "")

        # replace \\ with \
        string = string.replace("\\\\", "\\")

        # replace tfrac and dfrac with frac
        string = string.replace("tfrac", "frac")
        string = string.replace("dfrac", "frac")

        # remove \left and \right
        string = string.replace("\\left", "")
        string = string.replace("\\right", "")

        # Remove circ (degrees)
        string = string.replace("^{\\circ}", "")
        string = string.replace("^\\circ", "")

        # remove dollar signs
        string = string.replace("\\$", "")

        # remove units (on the right)
        string = self.remove_right_units(string)

        # remove percentage
        string = string.replace("\\%", "")
        string = string.replace("\%", "")

        # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
        string = string.replace(" .", " 0.")
        string = string.replace("{.", "{0.")
        # if empty, return empty string
        if len(string) == 0:
            return string
        if string[0] == ".":
            string = "0" + string

        # to consider: get rid of e.g. "k = " or "q = " at beginning
        if len(string.split("=")) == 2:
            if len(string.split("=")[0]) <= 2:
                string = string.split("=")[1]

        # fix sqrt3 --> sqrt{3}
        string = self.fix_sqrt(string)

        # remove spaces
        string = string.replace(" ", "")

        # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
        string = self.fix_fracs(string)

        # manually change 0.5 --> \frac{1}{2}
        if string == "0.5":
            string = "\\frac{1}{2}"

        # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
        string = self.fix_a_slash_b(string)

        return string


class MathAlgebra(Math):
    VERSION = 0
    def get_file_info(self):
        return 'algebra'


class MathCountingAndProbability(Math):
    VERSION = 0
    def get_file_info(self):
        return 'counting_and_probability'


class MathGeometry(Math):
    VERSION = 0
    def get_file_info(self):
        return 'geometry'


class MathIntermediateAlgebra(Math):
    VERSION = 0
    def get_file_info(self):
        return 'intermediate_algebra'


class MathNumberTheory(Math):
    VERSION = 0
    def get_file_info(self):
        return 'number_theory'


class MathPrealgebra(Math):
    VERSION = 0
    def get_file_info(self):
        return 'prealgebra'


class MathPrecalculus(Math):
    VERSION = 0
    def get_file_info(self):
        return 'precalculus'
