import re


environment_info = {
        "ladders": set(),
    }


def clear_environment_info():
    """
    Clears the environment information by resetting the ladders set.
    """
    global environment_info
    environment_info["ladders"].clear()


def reward_engineering(current_state: str, next_state: str) -> float:
    """
    Shapes a dictionary based on objects e.g. ladders and players, seen in the environment.

    Parameters:
    - current_state (str): The current state of the environment as a string.
    - next_state (str): The next state of the environment as a string.

    Returns:
    - dict: A dictionary containing ladders and players seen in the environment.
    """
    
    ladder_pattern = re.search(r"(leftLadder|rightLadder)\((\w+),(\w+)\)", next_state)
    on_ladder_pattern = re.search(r"onLadder\((\w+),(\w+)\)" , current_state)
    goal_pattern = re.search(r"sameLevelChild\((\w+),(\w+)\)" , next_state)

    

    if ladder_pattern and on_ladder_pattern and on_ladder_pattern.group(2)!= ladder_pattern.group(3):
        ladder_second_arg = on_ladder_pattern.group(2)

        if ladder_second_arg not in environment_info["ladders"]:
            environment_info["ladders"].add(ladder_second_arg)
            print(f"New ladder added: {ladder_second_arg}")
            print(f"Updated environment info: {environment_info}")
            return [10.0]

    elif on_ladder_pattern and goal_pattern:
        print(f"Goal reached with player {on_ladder_pattern.group(1)} on ladder {on_ladder_pattern.group(2)}")
        return [100.0]
    
    return [-1.0]