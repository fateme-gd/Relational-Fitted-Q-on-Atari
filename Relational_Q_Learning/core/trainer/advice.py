

from typing import List, Dict, Callable
import re
from typing import List, Dict, Tuple, Union
from collections import namedtuple

# Predicate = namedtuple("Predicate", ["name", "args"])

def parse_predicate(predicate_str: str) -> str:
    predicate_str = predicate_str.strip().rstrip('.').lower()
    name, args_str = predicate_str.split('(', 1)
    args = args_str[:-1].split(',')  # remove trailing ')'
    # return Predicate(name, [arg.strip() for arg in args])
    return name


class AdviceRule:
    def __init__(self, rule_body: List[str], preferred_action: str):
        self.rule_bodies = [parse_predicate(p) for p in rule_body]
        print(f"lifted_predicates: {self.rule_bodies}")
        self.preferred_action = preferred_action

    def applies_to(self, grounded_predicates: List[str]) -> bool:
        grounded = [parse_predicate(p) for p in grounded_predicates]
        matched = 0
        for body in self.rule_bodies:
            if body in grounded:
                matched += 1

        return matched == len(self.rule_bodies)

class AdviceHandler:
    def __init__(self, advice_rules: List[AdviceRule]):
        self.advice_rules = advice_rules

    def get_advised_action(self, grounded_state: List[str]) -> str:
        for rule in self.advice_rules:
            if rule.applies_to(grounded_state):
                return rule.preferred_action
        return None


# def select_action_with_advice(
#     state_predicates: List[str],
#     get_q_values: Callable[[List[str]], List[float]],
#     advice_handler: AdviceHandler
# ) -> str:
#     q_values = get_q_values(state_predicates)
    
#     max_q_action = max(q_values)

#     advised_action = advice_handler.get_advised_action(state_predicates)
#     print(f"q_values: {q_values}, advised_action: {advised_action}")

#     if advised_action and advised_action in q_values:
#         print(f"[Advice] Overriding '{max_q_action}' with advised '{advised_action}'")
#         return advised_action

#     return max_q_action
