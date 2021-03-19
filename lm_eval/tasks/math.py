import json
import os
from lm_eval.utils import sh
from lm_eval.metrics import mean
from lm_eval.base import Task, rf
import abc

class Math(Task):
    directory = 'data/MATH'
    def download(self):
        if not os.path.exists('data/'):
            sh("mkdir data")
        if not os.path.exists('data/MATH/'):
            sh("wget https://people.eecs.berkeley.edu/~hendrycks/MATH.tar.gz -P data/")
            sh("tar -xvf data/MATH.tar.gz -C data/")
            sh("rm data/MATH.tar.gz")
        self.set_docs()

    @abc.abstractmethod
    def get_file_info(self):
        """returns directory name"""
        pass

    def set_docs(self):
        self._training_docs = []
        self._testing_docs = []
        dir_name = self.get_file_info()
        train_path = os.path.join('data/MATH/train',dir_name).replace("\\", "/") 
        test_path = os.path.join('data/MATH/test',dir_name).replace("\\", "/")

        for filename in os.listdir(train_path):
            with open(os.path.join(train_path, filename).replace("\\", "/")) as f: 
                self._training_docs.append(json.load(f))
        for filename in os.listdir(test_path):
            with open(os.path.join(test_path, filename).replace("\\", "/")) as f: 
                self._testing_docs.append(json.load(f))

        for doc in self._testing_docs:
            doc["answer"] = self.remove_boxed(self.last_boxed_only_string(doc["solution"]))
        self._testing_docs = self._testing_docs[:25]


    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self._training_docs

    def validation_docs(self):
        return NotImplemented

    def test_docs(self):
        return self._testing_docs

    def fewshot_examples(self, k):
        assert k <= 8, "There are only 8 possible shots for this task."
        prompts = [
            {"problem": "What is $\left(\\frac{7}{8}\\right)^3 \cdot \left(\\frac{7}{8}\\right)^{-3}$?", "answer": "$1$"},
            {"problem": "In how many ways can 4 books be selected from a shelf of 6 books if the order in which the books are selected does not matter?", "answer": "$15$"},
            {"problem": "Find the distance between the points $(2,1,-4)$ and $(5,8,-3).$", "answer": "$\sqrt{59}$"},
            {"problem": "The faces of an octahedral die are labeled with digits $1$ through $8$. What is the probability, expressed as a common fraction, of rolling a sum of $15$ with a pair of such octahedral dice?", "answer": "$\\frac{1}{32}$"},
            {"problem": "The first three terms of an arithmetic sequence are 1, 10 and 19, respectively. What is the value of the 21st term?" , "answer": "$181$"},
            {"problem": "Calculate $6 \\cdot 8\\frac{1}{3}", "answer": "$50$"},
            {"problem": "When the binary number $100101110010_2$ is divided by 4, what is the remainder (give your answer in base 10)?", "answer": "$2$"},
            {"problem": "How many zeros are at the end of the product 25 $\\times$ 240?", "answer": "$3$"}
        ]
        return prompts[:k]
  
    def doc_to_text(self, doc):
        return  "\n" + "###" + "\n" + "Problem: " + doc["problem"] + "\n" + "Answer:"


    def doc_to_target(self, doc):
        return " " + doc["answer"]

    def construct_requests(self, doc, ctx):
        to_send = self.doc_to_text(doc)
        answer = self.doc_to_target(doc)
        ll = rf.greedy_until(ctx + to_send, "\n")
        return ll
    

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

    def fewshot_description(self):
        return "Given a mathematics problem, determine the answer. Simplify your answer as much as possible." + "\n"

    def fewshot_context(self, doc, num_fewshot, provide_description):
        description = self.fewshot_description()

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            labeled_examples = "\n\n".join(
                [self.doc_to_text(doc) + self.doc_to_target(doc) + "\n" + "###" + "\n" for doc in self.fewshot_examples(k=num_fewshot)]
            ) + "\n\n"

        return description + labeled_examples


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
        left = "\\boxed{"
        try:
            assert s[:len(left)] == left
            assert s[-1] == "}"
            return s[len(left):-1]
        except:
            return None

    def last_boxed_only_string(self, string):
        idx = string.rfind("\\boxed")
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
        
        if right_brace_idx == None:
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
                    except:
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
        except:
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
    def get_file_info(self):
        return 'algebra'

class MathCountingAndProbability(Math):
    def get_file_info(self):
        return 'counting_and_probability'

class MathGeometry(Math):
    def get_file_info(self):
        return 'geometry'

class MathIntermediateAlgebra(Math):
    def get_file_info(self):
        return 'intermediate_algebra'

class MathNumberTheory(Math):
    def get_file_info(self):
        return 'number_theory'

class MathPrealgebra(Math):
    def get_file_info(self):
        return 'prealgebra'

class MathPrecalculus(Math):
    def get_file_info(self):
        return 'precalculus'
