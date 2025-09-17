

import math
from typing import List, Dict, Callable
import re
from typing import List, Dict, Tuple, Union
from collections import namedtuple
import numpy as np
import numpy as np
import re

from collections import defaultdict, Counter


# Predicate = namedtuple("Predicate", ["name", "args"])

def parse_predicate(predicate_str: str) -> str:
    predicate_str = predicate_str.strip().rstrip('.').lower()
    name, args_str = predicate_str.split('(', 1)
    args = args_str[:-1].split(',')  # remove trailing ')'
    # return Predicate(name, [arg.strip() for arg in args])
    return name


class AdviceRule:
    def __init__(self, rule_body: List[str], preferred_action: str, enforce:bool = False):
        self.rule_bodies = [parse_predicate(p) for p in rule_body]
        print(f"lifted_predicates: {self.rule_bodies}")
        self.preferred_action = preferred_action
        self.enforce = enforce  

    def applies_to(self, grounded_predicates: List[str]) -> bool:
        grounded = [parse_predicate(p) for p in grounded_predicates]
        matched = 0

        # if not self.enforce:
        return set(self.rule_bodies) == set(grounded)
        
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

# Functions to calculate uncertainty

def remove_state_id(predicate: str) -> str:
    """
    Removes the state id from the end of a predicate string.
    Example: rightOfLadder(obj1,obj16,s0) -> rightOfLadder(obj1,obj16)
    """
    # Match pattern: function(args,someStateId)
    # return re.sub(r'(,\s*s\d+\s*\))', ')', predicate)
    return predicate.split('(')[0]

def extract_actions_from_examples(pos):
    """
    Extract action names from a list of regressionExample lines.
    
    Args:
        lines (list of str): Each line like 'regressionExample(up(obj1,s3),0.000).'
        
    Returns:
        List of action strings, e.g., ['noop(obj1,s0)', 'left(obj1,s1)', ...]
    """
    actions = []
    pattern = re.compile(r"regressionExample\((.*?\(.*?\)),\s*[-+]?[0-9]*\.?[0-9]+\)\.")
    
    for line in pos:
        match = pattern.match(line.strip())
        if match:
            actions.append(match.group(1))  # extract action(obj1,sX)
    
    return actions

def count_states(facts):
    stat = {}
    for fact in facts:
        fact = remove_state_id(fact)
        if fact not in stat:
            stat[fact] = 0
        stat[fact] += 1
    return stat

def compute_entropy(action_list):
        counts = Counter(action_list)
        total = sum(counts.values())
        probs = [count / total for count in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs)
        return entropy

def seek_advice(facts: List[str], positives: List[str]):
    count_stat = count_states(facts)
    actions = extract_actions_from_examples(positives)

    symbolic_state_action_map = defaultdict(list)

    for state_pred, action in zip(facts, actions):
        base_state = remove_state_id(state_pred)
        action_type = action.split('(')[0]  # extract action name only
        symbolic_state_action_map[base_state].append(action_type)

    for sym_state, acts in symbolic_state_action_map.items():
        entropy = compute_entropy(acts)
        print(f"State: {sym_state} -> Entropy: {entropy:.4f}, Actions: {Counter(acts)}")
    print("State counts:", count_stat)
    # print("state_action_map:", symbolic_state_action_map)


def compute_entropy_for_qtable(q_table, temperature=1.0):
    """
    Compute the normalized Shannon entropy for each state in a Q-table,
    treating equal Q-values (including all zeros) as a uniform distribution.
    
    Args:
        q_table (dict): {state: [q_value1, q_value2, ...], ...}
        temperature (float): Temperature for softmax normalization (default=1.0).
    
    Returns:
        dict: {state: normalized_entropy}
    """
    entropy_results = {}

    for state, q_values in q_table.items():
        # If all Q-values are identical, treat as uniform (max entropy)
        if len(set(q_values)) == 1:
            max_entropy = math.log(len(q_values), 2) if len(q_values) > 0 else 1
            entropy_results[state] = 1.0  # normalized max entropy
            continue

        # Otherwise, compute softmax probabilities
        exps = [math.exp(q / temperature) for q in q_values]
        sum_exps = sum(exps)
        probs = [e / sum_exps for e in exps]

        # Shannon entropy
        entropy = -sum(p * math.log(p, 2) for p in probs if p > 0)
        max_entropy = math.log(len(q_values), 2) if len(q_values) > 0 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        entropy_results[state] = normalized_entropy

    return entropy_results


def compute_uncertainty_stats(q_table, metric='std'):
    """
    metric: 'std' for standard deviation, 'range' for range between top two Q-values.
    """
    state_uncertainty = {}
    max_uncertainty = -np.inf
    most_uncertain_state = None

    for key, q_values in q_table.items():
        q_values = np.array(q_values)
        
        if metric == 'std':
            unc = np.std(q_values)
        elif metric == 'range':
            sorted_vals = np.sort(q_values)[::-1]
            unc = sorted_vals[0] - sorted_vals[1] if len(sorted_vals) > 1 else 0.0
        else:
            raise ValueError("Unsupported metric. Use 'std' or 'range'.")

        state_uncertainty[key] = unc

        if unc > max_uncertainty:
            max_uncertainty = unc
            most_uncertain_state = key

    # Convert to list for stats
    uncertainties = list(state_uncertainty.values())
    
    stats = {
        'mean': np.mean(uncertainties),
        'std': np.std(uncertainties),
        'max': np.max(uncertainties),
        'median': np.median(uncertainties)
    }

    return most_uncertain_state, stats


