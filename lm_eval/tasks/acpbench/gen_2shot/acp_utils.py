import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

from lm_eval.api.registry import register_filter
from lm_eval.filters.extraction import RegexFilter


try:
    import tempfile

    import tarski
    from kstar_planner import planners as kp
    from lark import Lark
    from lark.lexer import Token
    from lark.visitors import Visitor
    from pddl.core import Problem
    from pddl.parser.domain import DomainParser
    from pddl.parser.problem import ProblemParser
    from tarski.grounding.common import StateVariableLite
    from tarski.grounding.lp_grounding import LPGroundingStrategy
    from tarski.io import PDDLReader
    from tarski.io import fstrips as iofs
    from tarski.syntax.formulas import is_atom
    from tarski.syntax.transform.action_grounding import (
        ground_schema_into_plain_operator_from_grounding,
    )
    from tarski.util import SymbolIndex
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "`lark>=1.1.9`, `tarski[clingo]==0.8.2`, `pddl==0.4.2` and `kstar-planner==1.4.2` are required for evaluating the generative tasks. \
Please install via pip install lm-eval[acpbench] or pip install -e .[acpbench]",
    )


#########################################################################
# Grammar


GRAMMAR_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "acp_grammar.lark"
)


class ACPBench_Visitor(Visitor):
    def __init__(self) -> None:
        super().__init__()
        self.action_lists = None
        self.action_names = None
        self.progression_lists = None
        self.prog_lists = None
        self.indexes = None

    def action_list(self, tree):
        self.action_lists = []

    def prog_list(self, tree):
        if self.prog_lists is not None:
            self.progression_lists.append(self.prog_lists)
        self.prog_lists = []

    def progression_list(self, tree):
        self.progression_lists = []

    def action_none(self, tree):
        self.action_names = "None"

    def action_name(self, tree):
        act_name = "(" + "".join(tree.children[1:-1]) + ")"
        self.action_names = act_name
        if self.action_lists is not None:
            self.action_lists.append(act_name)
        if self.prog_lists is not None:
            self.prog_lists.append(act_name)

    def index(self, tree):
        self.indexes = "".join(tree.children)
        if not self.indexes.isnumeric():
            self.indexes = None


class ACPGrammarParser(object):
    def __init__(self, task) -> None:
        self.task = task
        with open(GRAMMAR_FILE) as f:
            grammar = f.read()
            self.acp_parser = Lark(grammar, start=task, parser="lalr")

    def parse(self, input, debug=False):
        def ignore_errors(e):
            if hasattr(e, "token") and e.token.type == "$END":
                for x in e.expected:
                    if x != "WS":
                        e.interactive_parser.feed_token(
                            Token(x, self.acp_parser.get_terminal(x).pattern.value)
                        )

            return True

        input = input.replace("\n", "")
        input = input.strip()
        try:
            tree = self.acp_parser.parse(input, on_error=ignore_errors)

            if debug:
                print(tree)
            visitor = ACPBench_Visitor()
            visitor.visit_topdown(tree)
            if self.task == "action_list":
                return visitor.action_lists
            elif self.task == "act":
                return visitor.action_names
            elif self.task == "action_name":
                return visitor.action_names
            elif self.task == "index":
                return visitor.indexes
            elif self.task == "progression_list":
                if visitor.prog_lists not in visitor.progression_lists:
                    visitor.progression_lists.append(visitor.prog_lists)
                return visitor.progression_lists
        except Exception as e:
            if debug:
                print("exception")
                print(e)
            return None


##############################################################################
# Utils


# Used in next action
def is_on_optimal_plan(domain, problem, action, opt):
    with (
        tempfile.NamedTemporaryFile() as domain_temp,
        tempfile.NamedTemporaryFile() as problem_temp,
    ):
        with open(str(domain_temp.name), "w", encoding="utf8") as file:
            file.write(domain.lower())
        with open(str(problem_temp.name), "w", encoding="utf8") as file:
            file.write(problem.lower())

        # Here, we need to keep the temp files live until the end of the function
        try:
            P = STRIPS(str(domain_temp.name), str(problem_temp.name))
        except Exception:
            # Unsolvable
            return False

        a = P.get_action_or_none(action[1:-1])
        if a is None:
            return False
        state = P.init
        next_state = progress(state, a)
        if opt is None:
            # Get an optimal plan cost
            plans = generate_optimal_plans_for_problem_state(
                P, state, num_plans=1, timeout=5
            )
            opt = len(plans[0]["actions"])
        else:
            opt = int(opt)

        # Getting an optimal plan for the next state
        next_plans = generate_optimal_plans_for_problem_state(
            P, next_state, num_plans=1, timeout=5
        )
        if next_plans is None:
            return False
        next_opt = len(next_plans[0]["actions"])
        return next_opt + 1 == opt


# Used in justification
def is_plan(domain, problem, new_plan):
    P = get_STRIPS(domain, problem)
    if P is None:
        # Unsolvable
        return False

    # Check if new_plan is a plan
    current_state = P.init
    for action in new_plan:
        applicable_actions = P.get_applicable_actions(current_state)
        app_actions_list = [f"({a.name.lower()})" for a in applicable_actions]
        if action.lower() not in app_actions_list:
            return False
        a = applicable_actions[app_actions_list.index(action.lower())]
        current_state = progress(current_state, a)
    return entails(current_state, P.goal)


# Used in action reachability
def get_action_preconditions(domain, problem, action):
    P = get_STRIPS(domain, problem)

    assert P is not None, f"Domain\n{domain}\nProblem\n{problem}\nAction: {action}"
    a = P.get_action_or_none(action[1:-1])
    if a is None:
        return a

    return [f"({f})" for f in a.pres]


def generate_optimal_plans_for_problem_state(P, state, num_plans, timeout):
    import tempfile

    with (
        tempfile.NamedTemporaryFile() as domain_temp,
        tempfile.NamedTemporaryFile() as problem_temp,
    ):
        create_tmp_dom_prob_replace_init(P, state, domain_temp, problem_temp)
        plans = generate_top_q_plans(
            domain=str(domain_temp.name),
            problem=str(problem_temp.name),
            num_plans=num_plans,
            quality_bound=1.0,
            timeout=timeout,
        )
        # print(plans)
        if plans is None or len(plans["plans"]) == 0:
            return None
        return plans["plans"]


def generate_top_q_plans(domain, problem, num_plans=10, quality_bound=1.0, timeout=30):
    # print("Running K* planner")
    plans = kp.plan_unordered_topq(
        domain_file=Path(domain),
        problem_file=Path(problem),
        number_of_plans_bound=num_plans,
        quality_bound=quality_bound,
        timeout=timeout,
    )
    return plans


# Used in (action) reachability
def is_unsolvable_new_goal(domain, problem, new_goal):
    goal = extract_goal(problem)
    new_problem = problem.replace(goal, f"(:goal {new_goal} )")
    return is_unsolvable(domain, new_problem)


def is_unsolvable(domain, problem):
    with (
        tempfile.NamedTemporaryFile() as domain_temp,
        tempfile.NamedTemporaryFile() as problem_temp,
    ):
        with open(str(domain_temp.name), "w", encoding="utf8") as file:
            file.write(str(domain))
        with open(str(problem_temp.name), "w", encoding="utf8") as file:
            file.write(str(problem))

        plans = kp.plan_unordered_topq(
            domain_file=Path(str(domain_temp.name)),
            problem_file=Path(str(problem_temp.name)),
            quality_bound=1.0,
            number_of_plans_bound=1,
            timeout=3,
        )

        if len(plans["planner_error"]) > 0:
            fl = plans["planner_error"].split("\n")[0]
            print(f"Planner error: {fl}")
            return False
        if plans is None or len(plans["plans"]) == 0:
            return plans["unsolvable"]
        return False


def extract_goal(prob):
    a = prob.split("(:goal")[1]
    cp = 1
    for i, c in enumerate(a):
        if c == ")":
            cp -= 1
        if c == "(":
            cp += 1
        if cp == 0:
            return "(:goal" + a[: i + 1]

    assert False


def entails(state, partialstate):
    return partialstate <= state


def progress(state, act):
    assert entails(state, act.pres), (
        "Cannot progress with inconsistent state / action precondition:\n\t Action: "
        + act.name
        + "\n\t State: \n\t\t"
        + "\n\t\t".join(state)
    )
    return (state - act.dels) | act.adds


def regress(state, act):
    assert len(state & act.dels) == 0, (
        "Cannot regress with inconsistent state / action delete effect:\n\t Action: "
        + act.name
        + "\n\t State: \n\t\t"
        + "\n\t\t".join(state)
    )
    return (state - act.adds) | act.pres


def get_STRIPS(domain, problem):
    with (
        tempfile.NamedTemporaryFile() as domain_temp,
        tempfile.NamedTemporaryFile() as problem_temp,
    ):
        with open(str(domain_temp.name), "w", encoding="utf8") as file:
            file.write(domain.lower())
        with open(str(problem_temp.name), "w", encoding="utf8") as file:
            file.write(problem.lower())

        try:
            P = STRIPS(str(domain_temp.name), str(problem_temp.name))
            return P
        except Exception as e:
            print(f"||{e}||")
            return None


def create_tmp_dom_prob_replace_init(P, state, result_domain_file, result_problem_file):
    d, p = P.PDDL_replace_init_pddl_parser(state)
    with open(str(result_domain_file.name), "w", encoding="utf8") as file:
        file.write(str(d))
    with open(str(result_problem_file.name), "w", encoding="utf8") as file:
        file.write(str(p))

    return d, p


def fix_name(s):
    # (act param)
    if "(" == s[0] and ")" == s[-1]:
        return s[1:-1]
    # make it space separated
    s = s.replace(", ", " ").replace(",", " ")
    # act(param)
    if "(" in s:
        assert ")" == s[-1], f"Broken name? {s}"
        s = s.replace("(", " ").replace(")", "")
    # act param
    return s


def get_atoms_pddl(d, p, atoms):
    objs = set()
    preds = defaultdict(list)
    for atom in atoms:
        a = atom.lower().strip().split(" ")
        args = a[1:]
        preds[a[0]].append(args)
        objs |= set(args)

    constants = [o for o in p.objects | d.constants if o.name.lower() in objs]
    constants_dict = {}
    for c in constants:
        constants_dict[c.name.lower()] = c
    assert len(objs) == len(constants), (
        f"Could not identify all objects: {objs - set(constants_dict.keys())} not found, {set(constants_dict.keys()) - objs} should not be there"
    )

    state = []
    covered_preds = set()
    for f in d.predicates:
        name = f.name.lower()
        if name in preds:
            covered_preds.add(name)
            assert len(preds[name][0]) == f.arity, (
                f"The arity does not match: {preds[name]} vs {f.terms}"
            )
            # Going over the lists of objects, adding ground predicate for each
            for ob in preds[name]:
                c = [constants_dict[o] for o in ob]
                state.append(f(*c))
    assert len(covered_preds) == len(preds.keys()), (
        f"Covered predicates: \n{sorted(list(covered_preds))} vs \n{sorted(list(preds.keys()))}"
    )
    return set(state)


class Action:
    def __init__(self, name, pre, add, delete):
        self.name = name
        self.pres = pre
        self.adds = add
        self.dels = delete

    def __str__(self):
        pres = "{" + ", ".join([f"({a})" for a in self.pres]) + "}"
        adds = "{" + ", ".join([f"({a})" for a in self.adds]) + "}"
        dels = "{" + ", ".join([f"({a})" for a in self.dels]) + "}"

        return f"< {self.name}, {pres}, {adds}, {dels} >"

    def toJSON(self):
        return json.dumps(
            {
                "name": self.name,
                "preconditions": [f"({a})" for a in self.pres],
                "add_effects": [f"({a})" for a in self.adds],
                "delete_effects": [f"({a})" for a in self.dels],
            },
            sort_keys=True,
            indent=4,
        )

    def __repr__(self):
        return self.name

    def __eq__(self, action):
        return self.name == action.name

    def __hash__(self):
        return hash(self.name)


class STRIPS:
    def __init__(self, domain, problem):
        self.domain_file = domain
        self.problem_file = problem
        self.reader = PDDLReader(raise_on_error=True)
        self.reader.parse_domain(domain)
        self.problem = self.reader.parse_instance(problem)
        (self.grounded_fluents, init, goal, self.operators, self.grounder) = (
            self.ground_problem(self.problem)
        )

        self.fluents = set([fix_name(str(f)) for f in self.grounded_fluents])
        self.fluents_map = dict()
        for f in self.grounded_fluents:
            self.fluents_map[fix_name(str(f))] = f
        self.init = set([fix_name(str(f)) for f in init])
        self.goal = set([fix_name(str(f)) for f in goal])
        self.actions = set()
        self.action_map = {}
        self.init_fluents = [self.fluents_map[f] for f in self.init]

        self.static_predicates = [i.name for i in self.grounder.static_symbols]
        for op in self.operators:
            act = self.operator_to_action(op)
            self.actions.add(act)
            self.action_map[act.name.lower()] = act

    def __str__(self):
        fluents = "P = {" + ", ".join([f"({a})" for a in self.fluents]) + "}"
        init = "I = {" + ", ".join([f"({a})" for a in self.init]) + "}"
        goal = "G = {" + ", ".join([f"({a})" for a in self.goal]) + "}"
        actions = "A = {" + "\n ".join([a.__str__() for a in self.actions]) + "}"
        return fluents + ",\n" + init + "\n" + goal + "\n" + actions

    def toJSON(self):
        actions = [a.toJSON() for a in self.actions]
        return json.dumps(
            {
                "fluents": list(self.fluents),
                "initial_state": list(self.init),
                "goal": list(self.goal),
                "actions": actions,
            },
            sort_keys=True,
            indent=4,
        )

    def operator_to_action(self, op, check_fluents=True, check_static=False):
        adds = {
            fix_name(str(f.atom)) for f in op.effects if isinstance(f, iofs.AddEffect)
        } & self.fluents
        dels = {
            fix_name(str(f.atom)) for f in op.effects if isinstance(f, iofs.DelEffect)
        } & self.fluents
        pre = self.fix_pre_name(op.precondition)
        if check_fluents:
            pre = pre & self.fluents
        if check_static:
            pre = {p for p in pre if p.split()[0] not in self.static_predicates}
        act = Action(fix_name(str(op)), pre, adds, dels)
        return act

    def fix_pre_name(self, precondition):
        if not is_atom(precondition):
            return {fix_name(str(f)) for f in precondition.subformulas}
        return {fix_name(str(precondition))}

    def action(self, name):
        return self.action_map[fix_name(name).lower()]

    def get_action_or_none(self, name):
        if "(" in name and ")" != name[-1]:
            return None
        return self.action_map.get(fix_name(name).lower(), None)

    def fluent(self, name):
        return fix_name(name)

    def static_symbols(self):
        return list(self.grounder.static_symbols)

    def fluent_symbols(self):
        return list(self.grounder.fluent_symbols)

    def get_grounded_atoms(self, symbol):
        variables = SymbolIndex()
        lang = symbol.language
        key = "atom_" + symbol.name
        model = self.grounder._solve_lp()
        if (
            key in model
        ):  # in case there is no reachable ground state variable from that fluent symbol
            for binding in model[key]:
                binding_with_constants = tuple(lang.get(c) for c in binding)
                variables.add(StateVariableLite(symbol, binding_with_constants))
        return variables

    def get_applicable_actions(self, s):
        return [a for a in self.actions if entails(s, a.pres)]

    def ground_problem(self, problem):
        grounder = LPGroundingStrategy(problem, include_variable_inequalities=True)
        action_groundings = grounder.ground_actions()
        operators = []
        for action_name, groundings in action_groundings.items():
            action = problem.get_action(action_name)
            for grounding in groundings:
                operators.append(
                    ground_schema_into_plain_operator_from_grounding(action, grounding)
                )

        grounded_fluents = set(
            [
                grounded_fluent.to_atom()
                for grounded_fluent in grounder.ground_state_variables().objects
            ]
        )
        init = [f for f in problem.init.as_atoms() if f in grounded_fluents]
        if isinstance(problem.goal, tarski.syntax.Atom):
            goal = [problem.goal]
        else:
            goal = [f for f in problem.goal.subformulas if f in grounded_fluents]

        return (grounded_fluents, init, goal, operators, grounder)

    def get_static(self):
        static_symbols = self.static_symbols()
        ret = []
        for symbol in static_symbols:
            ret.extend(self.get_grounded_atoms(symbol))
        return set([fix_name(str(x)) for x in ret])

    def PDDL_replace_init_pddl_parser(self, s):
        d = DomainParser()(open(self.domain_file, "r").read().lower())
        p = ProblemParser()(open(self.problem_file, "r").read().lower())

        new_state = get_atoms_pddl(d, p, s | self.get_static())

        new_p = Problem(
            p.name, domain=d, objects=p.objects, init=new_state, goal=p.goal
        )

        return d, new_p


def parse_ans(response: str, parser: ACPGrammarParser, task: str):
    return [parser.parse(clean_answer(resp, task)) for resp in response]


# def parse_ans(response : str, parser : ACPGrammarParser, task : str):
#     ans = [parser.parse(clean_answer(resp, task), debug=True) for resp in response]
#     if any(elem is None for elem in ans) or any(elem is None for elem in ans[0]):
#         return None
#     return ans


def remove_garbage(s):
    while True:
        if s.endswith("."):
            s = s[:-1]
        elif s.endswith("\n"):
            s = s[:-2]
        else:
            break
    return s.rstrip()


def compare_str(s1, s2):
    return remove_garbage(s1).lower() == remove_garbage(s2).lower()


def compare(l1, l2):
    if not isinstance(l1, list):
        return compare_str(l1, l2)
    if not isinstance(l2, list):
        return False
    for i, v in enumerate(l1):
        if not compare(v, l2[i]):
            return False
    return True


def check_prog_response(resp):
    if (
        "Positive Effects".lower() in resp.lower()
        and "Negative Effects".lower() in resp.lower()
    ):
        if "[" not in resp:
            return True
    return False


def clean_answer(resp, task):
    # Minor cleanup
    if "progression_gen" in task:
        # Check for Positive Effects and Negative Effects instead of separation
        if check_prog_response(resp):
            # replace **Positive Effects** with "["
            # replace **Negative Effects** with "] ["
            # append "]" to the end
            resp2 = resp.lower()
            resp2 = resp2.replace("*", "")
            resp2 = resp2.replace("positive effects", "[")
            resp2 = resp2.replace("negative effects", "] [")
            resp2 = resp2 + "]"
            return resp2
    if "action_justification_gen" in task:
        # Check for "simplified plan:"
        if "simplified plan:" in resp.lower():
            resp2 = resp.lower()
            resp2 = resp2.replace("*", "")
            resp2 = resp2.split("simplified plan:")[1]
            return resp2
    return resp


def get_grammar_task(task):
    # print(task)
    if task == "reachable_atom_gen":
        return "act"
    elif task == "progression_gen":
        return "progression_list"
    elif task == "validation_gen":
        return "index"
    elif task == "reachable_action_gen":
        return "act"
    elif task == "action_justification_gen":
        return "action_list"
    elif task == "landmarks_gen":
        return "act"
    elif task == "goal_closer_gen":
        return "action_name"
    elif task == "applicable_actions_gen":
        return "action_list"


##############################################################################
#  Evaluators


def fix_action_name(a):
    assert a.startswith("(") and a.endswith(")")
    return "(" + " ".join([x.strip() for x in a[1:-1].split(" ") if len(x) > 0]) + ")"


def str_remove_before_first_parentheses(s):
    if s.startswith("("):
        return s
    try:
        return s[s.index("(") :]
    except Exception:
        return ""


def str_remove_after_last_parentheses(s):
    if s.endswith(")"):
        return s

    i = s.rfind(")")

    if i == -1:
        return ""
    return s[: i + 1]


def cleanup_answer(ans):
    if isinstance(ans, str):
        ans = str_remove_before_first_parentheses(ans)
        ans = str_remove_after_last_parentheses(ans)
        ans = ans.lower()
        ans = (
            ans.replace(")\n(", ")######(")
            .replace("),(", ")######(")
            .replace(") (", ")######(")
            .split("######")
        )
        return ans
    if isinstance(ans, list):
        res = []
        for x in ans:
            res.extend(cleanup_answer(x))
        return res


def set_equal(ans1, ans2):
    return set(ans1) == set(ans2)


class BaseEvaluator(ABC):
    def __init__(self) -> None:
        self.scores = []

    @abstractmethod
    def get_score(self, ans, doc):
        pass

    def add_scores(self, scores):
        self.scores.extend(scores)

    def get_avg_score(self):
        avg_score = sum(self.scores) / len(self.scores)
        return avg_score


def get_evaluator(group):
    if group == "applicable_actions_gen":
        return ApplicabilityEvaluator()
    elif group == "progression_gen":
        return ProgressionEvaluator()
    elif group == "validation_gen":
        return ValidationEvaluator()
    elif group == "reachable_atom_gen":
        return ReachabilityEvaluator()
    elif group == "goal_closer_gen":
        return NextActionEvaluator()
    elif group == "action_justification_gen":
        return JustificationEvaluator()
    elif group == "landmarks_gen":
        return LandmarksEvaluator()
    elif group == "reachable_action_gen":
        return ActionReachabilityEvaluator()
    assert True, f"Group {group} not found"


"""
Action Reachability task: generate a valid action that is not applicable to any reachable state.
answer: A subset of actions that are known to be unreachable (not an exhaustive set).
        It is empty only when we *know* that there are no such actions.
"""


class ActionReachabilityEvaluator(BaseEvaluator):
    def get_score(self, ans, doc):
        real_answer = doc["answer"]
        if not real_answer or len(real_answer) == 0:
            # The correct answer is None
            self.add_scores(
                ["none" == x.strip().lower() if x is not None else False for x in ans]
            )
        else:
            for x in ans:
                if x is None:
                    self.scores.append(False)
                    continue
                action = x.strip().lower()
                if action in real_answer:
                    # The answer is in the subset of stored correct answers
                    self.scores.append(True)
                    continue
                prec = get_action_preconditions(
                    doc["PDDL_domain"].lower(), doc["PDDL_problem"].lower(), action
                )
                if prec is None:
                    # The answer does not correspond to a valid action
                    self.scores.append(False)
                else:
                    # Need to run a planner on a task with the answer action preconditions as the new goal
                    prec = f"(and {' '.join(prec)})"
                    self.scores.append(
                        is_unsolvable_new_goal(
                            doc["PDDL_domain"].lower(),
                            doc["PDDL_problem"].lower(),
                            prec,
                        )
                    )

        return self.get_avg_score()


"""
Action Applicability task: generate all actions that are applicable in the current state.
answer: A set of all applicable actions.
"""


class ApplicabilityEvaluator(BaseEvaluator):
    def get_score(self, ans, doc):
        real_answer = doc["answer"]
        real_answer = [a.lower() for a in real_answer]
        ans = [[fix_action_name(a) for a in x] if x is not None else None for x in ans]

        # Check if the answer is equal (as a set) to the real stored answer
        self.add_scores(
            [
                set_equal(real_answer, cleanup_answer(x)) if x is not None else False
                for x in ans
            ]
        )
        return self.get_avg_score()


def is_subsequence(plan, new_plan):
    i = 0
    for a in plan:
        if a == new_plan[i]:
            i += 1
            if len(new_plan) == i:
                # Done
                return True
    return False


def is_subsequence_and_plan(domain, problem, plan, new_plan):
    if len(plan) <= len(new_plan):
        return False
    if not is_subsequence(plan, new_plan):
        return False
    return is_plan(domain, problem, new_plan)


"""
Justification task: generate a proper subsequence of the given plan that is also a plan.
answer: A list of examples of actions that can be removed (ignored in evaluation).
"""


class JustificationEvaluator(BaseEvaluator):
    def get_score(self, ans, doc):
        # Sequence of actions (plan) from the question
        if "inputs" in doc:  # old field name
            seq = doc["inputs"][19:-147]
        else:
            seq = doc["question"][19:-147]
        seq = seq.replace(") (", ")######(").split("######")
        for x in ans:
            if x is None:
                self.scores.append(False)
                continue
            # An answer plan candidate
            x = [fix_action_name(a) for a in x]
            if len(x) == 0:
                # Wrong answer - never an empty sequence
                self.scores.append(0)
                continue
            # Check if the plan candidate from the answer (a) is a proper subsequence of the plan in the question and (b) is a plan.
            self.scores.append(
                is_subsequence_and_plan(
                    doc["PDDL_domain"].lower(), doc["PDDL_problem"].lower(), seq, x
                )
            )
        return self.get_avg_score()


"""
Landmarks task: generate a fact that is a non-trivial landmark for the current state.
answer: A list of facts that are found to be landmarks and a list of facts that are found to be non-landmarks.

The questions are generated only for cases where all facts either
    (a) hold in the current state,
    (b) true in goal,
    (c) are found to be landmarks, or
    (d) are found to be non-landmarks.
In such cases, the evaluation is simple, it does not require checking whether a fact is a landmark, it was
already done during question generation.
"""


class LandmarksEvaluator(BaseEvaluator):
    def get_score(self, ans, doc):
        # The set of facts that are found to be landmarks
        real_answer = doc["answer"]
        real_answer_yes = [a.lower() for a in real_answer["yes"]]

        for x in ans:
            if x is None:
                self.scores.append(False)
                continue
            if x.strip().lower() in real_answer_yes:
                # The answer fact is known to be landmark
                self.scores.append(True)
            elif x.strip().lower() == "none":
                # The answer is none, correct only if there are no known landmarks,
                #   since we only generate questions when that means that there are no non-trivial landmarks
                self.scores.append(len(real_answer_yes) == 0)
            else:
                # All other cases the answer is incorrect
                self.scores.append(False)

        return self.get_avg_score()


"""
Next Action task: generate an action that takes us closer to the goal.
answer:
    (a) A list of applicable actions that are known to be correct answers
    (b) A list of applicable actions that are known to be incorrect answers
    (c) The rest of the applicable actions (maybe).
"""


class NextActionEvaluator(BaseEvaluator):
    def get_score(self, ans, doc):
        real_answer = doc["answer"]
        real_answer_yes = [a.lower() for a in real_answer["yes"]]
        real_answer_no = [a.lower() for a in real_answer["no"]]
        real_answer_maybe = [a.lower() for a in real_answer["maybe"]]
        # The cost of the optimal plan from the current state
        opt = real_answer.get("opt", None)
        for x in ans:
            if x is None:
                self.scores.append(False)
                continue
            action = x.strip().lower()
            if action in real_answer_yes:
                # Known to be correct
                self.scores.append(True)
            elif action in real_answer_no:
                # Known to be incorrect
                self.scores.append(False)
            elif action not in real_answer_maybe:
                # Not applicable, must be incorrect
                self.scores.append(False)
            else:
                # Unknown, need to run a planner to check whether the state that results from applying the action is closer to the goal
                #  meaning has smaller optimal plan cost.
                self.scores.append(
                    is_on_optimal_plan(
                        doc["PDDL_domain"].lower(),
                        doc["PDDL_problem"].lower(),
                        action,
                        opt,
                    )
                )

        return self.get_avg_score()


"""
Progression task: generate the positive and negative effects of an action in the current state.
answer:
    (a) A list of facts that were false and become true, when the action is applied
    (b) A list of facts that were true and become false, when the action is applied
"""


class ProgressionEvaluator(BaseEvaluator):
    def get_score(self, ans, doc):
        real_answer = doc["answer"]
        real_answer_pos = [a.lower() for a in real_answer["pos"]]
        real_answer_neg = [a.lower() for a in real_answer["neg"]]

        for x in ans:
            # The answer should be two lists. We allow for a single list and assume that the second one is empty (relaxed evaluation).
            if x is None or len(x) > 2 or len(x) < 1:
                self.scores.append(False)
            else:
                p = cleanup_answer(x[0])
                if len(x) == 2:
                    n = cleanup_answer(x[1])
                else:
                    # Assuming the last element is dropped because it is empty
                    n = []
                # Check if the answer is equal as sets to the correct answers.
                ans = [set_equal(real_answer_pos, p), set_equal(real_answer_neg, n)]
                self.scores.append(all(ans))

        return self.get_avg_score()


"""
Reachability task: generate a valid fact that will never become true in any reachable state.
answer: A subset of facts that are known to be unreachable (not an exhaustive set).
        It is empty only when we *know* that there are no such facts.
"""


class ReachabilityEvaluator(BaseEvaluator):
    def get_score(self, ans, doc):
        real_answer = doc["answer"]
        real_answer = [f"({x.strip().lower()})" for x in real_answer]

        if len(real_answer) == 0:
            # The correct answer is None
            self.add_scores(
                ["none" == x.strip().lower() if x is not None else False for x in ans]
            )
        else:
            for x in ans:
                if x is None:
                    self.scores.append(False)
                elif x.strip().lower() in real_answer:
                    # The answer is in the subset of stored correct answers
                    self.scores.append(True)
                else:
                    # Need to run a planner on a task with the answer fact as the new goal
                    atom = x.strip().lower()
                    self.scores.append(
                        is_unsolvable_new_goal(
                            doc["PDDL_domain"].lower(),
                            doc["PDDL_problem"].lower(),
                            atom,
                        )
                    )

        return self.get_avg_score()


"""
Validation task: generate an index of the first inapplicable action in the given sequence.
answer: the correct index.
"""


class ValidationEvaluator(BaseEvaluator):
    def get_score(self, ans, doc):
        real_answer = str(doc["answer"])
        assert int(real_answer) >= 0, (
            f"The index must be non-negative, received {real_answer}"
        )
        # Exact match
        self.add_scores(
            [
                real_answer.lower() == x.strip().lower() if x is not None else False
                for x in ans
            ]
        )

        return self.get_avg_score()


##############################################################################


def dump_item(item, **kwargs):
    return json.dumps(item)


def parse_prediction(prediction):
    try:
        ans = json.loads(prediction.strip())
        response = ans.get("answer", None)
        return response
    except Exception as e:
        print(f"Exception occurred {e}")
        return prediction


@register_filter("ACP_grammar_filter")
class ACPGrammarFilter(RegexFilter):
    """Filtering Index using"""

    def __init__(self, *args, **kwargs):
        self.parser = ACPGrammarParser(kwargs["grammar_task"])
        self.clean = kwargs["clean"] if "clean" in kwargs else None

    def clean_pos_neg(self, resp):
        # Check for Positive Effects and Negative Effects instead of separation
        if check_prog_response(resp):
            resp2 = resp.lower()
            resp2 = resp2.replace("*", "")
            resp2 = resp2.replace("positive effects", "[")
            resp2 = resp2.replace("negative effects", "] [")
            resp2 = resp2 + "]"
            return resp2
        return resp

    def clean_simplified_plan(self, resp):
        # Check for "simplified plan:"
        if "simplified plan:" in resp.lower():
            resp2 = resp.lower()
            resp2 = resp2.replace("*", "")
            resp2 = resp2.split("simplified plan:")[1]
            return resp2
        return resp

    def apply(self, resps, docs):
        if self.clean == "pos_neg":
            filtered_resps = [
                [self.parser.parse(self.clean_pos_neg(r)) for r in resp]
                for resp in resps
            ]
        elif self.clean == "simplified plan":
            filtered_resps = [
                [self.parser.parse(self.clean_simplified_plan(r)) for r in resp]
                for resp in resps
            ]
        else:
            filtered_resps = [[self.parser.parse(r) for r in resp] for resp in resps]
        return filtered_resps


def process_acp_results(doc, results):
    return {"score": get_evaluator(doc["group"]).get_score(results, doc)}


def get_score(references, predictions, **kwargs):
    # print(f"References: {references}")
    # print(f"Predictions: {predictions}")
    data = json.loads(references[0].strip())
    real_ans = data["answer"]
    task = data["group"]

    responses = [parse_prediction(prediction) for prediction in predictions]

    print(f"Real answer: {real_ans}")
    print(f"Model answers: {responses}")
    parser = ACPGrammarParser(get_grammar_task(task))
    ans = parse_ans(responses, parser, task)

    print(f"Parsed model answers: {ans}")
    score = get_evaluator(task).get_score(ans, data)

    return {"get_score": score}
