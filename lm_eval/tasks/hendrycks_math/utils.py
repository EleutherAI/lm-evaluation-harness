import datasets


def doc_to_text(doc):
    train_prompt = [
        "Problem:\nFind the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.\n\nSolution:\nThe expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct."
    ]
    train_prompt += [
        "Problem:\nIf $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$\n\nSolution:\nWe have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct."
    ]
    train_prompt += [
        "Problem:\nTerrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?\n\nSolution:\nIf Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$: \\begin{align*}\n30n&=480\\\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}'}\nFinal Answer: The final answer is $16$. I hope it is correct."
    ]
    train_prompt += [
        "Problem:\nIf the system of equations\n\n\\begin{align*}\n2x-y&=a,\\\\\n3y-6x &=b.\n\\end{align*}has a solution, find $\\frac{a}{b},$ assuming $b \\neq 0.$\n\nSolution:\nIf we multiply the first equation by $-3$, we obtain\n\n$$3y-6x=-3a.$$Since we also know that $3y-6x=b$, we have\n\n$$-3a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{1}{3}}.$$\nFinal Answer: The final answer is $\\frac{2}{3}$. I hope it is correct."
    ]
    train_prompt = "\n\n".join(train_prompt)
    return train_prompt + "\n\n" + "Problem:\n" + doc["problem"] + "\n\n" + "Solution:"


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        out_doc = {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "final_solution": remove_boxed(last_boxed_only_string(doc["solution"])),
        }
        return out_doc

    return dataset.map(_process_doc)


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
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
        retval = string[idx : right_brace_idx + 1]

    return retval
