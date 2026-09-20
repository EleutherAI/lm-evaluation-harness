from lm_eval.tasks.hendrycks_math.utils import process_results


def test_process_results_accepts_multiple_boxed_answers():
    doc = {"solution": r"Thus, the answer is $\boxed{3, 5, 7}.$"}
    result = r"""### Final Answer

$$
\boxed{3}, \boxed{5}, \boxed{7}
$$"""

    assert process_results(doc, [result]) == {"exact_match": 1}


def test_process_results_accepts_unordered_simple_answer_lists():
    doc = {"solution": r"Thus, the answer is $\boxed{1,-2}.$"}
    result = r"""Final answer:

$$
\boxed{-2}, \boxed{1}
$$"""

    assert process_results(doc, [result]) == {"exact_match": 1}


def test_process_results_strips_display_math_delimiters():
    doc = {
        "solution": r"Thus, the answer is $\boxed{\begin{pmatrix} -1/3 \\ 2/3 \\ 5/3 \end{pmatrix}}.$"
    }
    result = r"""### Final Answer

$$
\boxed{\begin{pmatrix} -\dfrac{1}{3} \\ \dfrac{2}{3} \\ \dfrac{5}{3} \end{pmatrix}}
$$"""

    assert process_results(doc, [result]) == {"exact_match": 1}


def test_process_results_accepts_base_suffix_omitted_from_boxed_answer():
    doc = {"solution": r"Thus, the answer is $\boxed{52_8}.$"}
    result = r"""Thus, the product is:

$$
\boxed{52}
$$"""

    assert process_results(doc, [result]) == {"exact_match": 1}
