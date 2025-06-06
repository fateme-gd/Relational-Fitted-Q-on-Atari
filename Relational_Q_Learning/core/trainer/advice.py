

from typing import List, Dict, Callable
import re
from typing import List, Dict, Tuple, Union
from collections import namedtuple

Predicate = namedtuple("Predicate", ["name", "args"])

def parse_predicate(predicate_str: str) -> Predicate:
    predicate_str = predicate_str.strip().rstrip('.').lower()
    name, args_str = predicate_str.split('(', 1)
    args = args_str[:-1].split(',')  # remove trailing ')'
    return Predicate(name, [arg.strip() for arg in args])

def unify(lifted: Predicate, grounded: Predicate) -> Union[Dict[str, str], None]:
    if lifted.name != grounded.name or len(lifted.args) != len(grounded.args):
        return None
    bindings = {}
    for l_arg, g_arg in zip(lifted.args, grounded.args):
        if re.match(r'^[a-z]\w*$', l_arg):  # lifted var (e.g., x, p)
            bindings[l_arg] = g_arg
        elif l_arg != g_arg:
            return None
    return bindings


class AdviceRule:
    def __init__(self, lifted_conditions: List[str], preferred_action: str):
        self.lifted_predicates = [parse_predicate(p) for p in lifted_conditions]
        self.preferred_action = preferred_action

    def applies_to(self, grounded_predicates: List[str]) -> bool:
        grounded = [parse_predicate(p) for p in grounded_predicates]

        matched = 0
        for lifted_pred in self.lifted_predicates:
            for g_pred in grounded:
                if unify(lifted_pred, g_pred) is not None:
                    matched += 1
                    break  # one match is enough per lifted predicate

        return matched == len(self.lifted_predicates)

class AdviceHandler:
    def __init__(self, advice_rules: List[AdviceRule]):
        self.advice_rules = advice_rules

    def get_advised_action(self, grounded_state: List[str]) -> str:
        for rule in self.advice_rules:
            if rule.applies_to(grounded_state):
                return rule.preferred_action
        return None


def select_action_with_advice(
    state_predicates: List[str],
    get_q_values: Callable[[List[str]], List[float]],
    advice_handler: AdviceHandler
) -> str:
    q_values = get_q_values(state_predicates)
    max_q_action = max(q_values)

    advised_action = advice_handler.get_advised_action(state_predicates)

    if advised_action and advised_action in q_values:
        print(f"[Advice] Overriding '{max_q_action}' with advised '{advised_action}'")
        return advised_action

    return max_q_action
