import re



def clear_environment_info():
    """
    Clears the environment information by resetting the ladders set.
    """
    environment_info = {
        "ladders": set(),
    }

    return environment_info


def reward_engineering(environment_info, current_state: str, next_state: str) -> float:
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

    reward = [-1.0]    

    if ladder_pattern and on_ladder_pattern and on_ladder_pattern.group(2)!= ladder_pattern.group(3):
        ladder_second_arg = on_ladder_pattern.group(2)

        if ladder_second_arg not in environment_info["ladders"]:
            environment_info["ladders"].add(ladder_second_arg)
            print(f"New ladder added: {ladder_second_arg}")
            print(f"Updated environment info: {environment_info}")
            reward = [10.0]
            return [10.0], environment_info

    elif on_ladder_pattern and goal_pattern:
        print(f"Goal reached with player {on_ladder_pattern.group(1)} on ladder {on_ladder_pattern.group(2)}")
        reward = [100.0]
    
    return reward, environment_info


    # left_ladder_pattern = re.search(r"leftLadder\((\w+),(\w+)\)", current_state)
    # right_ladder_patter = re.search(r"rightLadder\((\w+),(\w+)\)", current_state)

    # left_ladder_pattern_next = re.search(r"leftLadder\((\w+),(\w+)\)", next_state)
    # right_ladder_patter_next = re.search(r"rightLadder\((\w+),(\w+)\)", next_state)

    # on_ladder_pattern = re.search(r"onLadder\((\w+),(\w+)\)" , current_state)
    # goal_pattern = re.search(r"sameLevelChild\((\w+),(\w+)\)" , next_state)

    # reward = [-1.0]    

    # if left_ladder_pattern :
    #     return -player.x , environment_info #inherently has step cost

    # elif right_ladder_patter:
    #     return player.x,    environment_info #inherently has step cost
    
    # if on_ladder_pattern and left_ladder_pattern_next and on_ladder_pattern.group(2)!= left_ladder_pattern_next.group(2):
    #     ladder_second_arg = on_ladder_pattern.group(2)

    #     if ladder_second_arg not in environment_info["ladders"]:
    #         environment_info["ladders"].add(ladder_second_arg)
    #         print(f"New ladder added: {ladder_second_arg}")
    #         print(f"Updated environment info: {environment_info}")
    #         reward = [10.0]
    #         return [10.0], environment_info
        
    # elif on_ladder_pattern and right_ladder_patter_next and on_ladder_pattern.group(2)!= right_ladder_patter_next.group(2):
    #     ladder_second_arg = on_ladder_pattern.group(2)

    #     if ladder_second_arg not in environment_info["ladders"]:
    #         environment_info["ladders"].add(ladder_second_arg)
    #         print(f"New ladder added: {ladder_second_arg}")
    #         print(f"Updated environment info: {environment_info}")
    #         reward = [10.0]
    #         return [10.0], environment_info

    # elif on_ladder_pattern and goal_pattern:
    #     print(f"Goal reached with player {on_ladder_pattern.group(1)} on ladder {on_ladder_pattern.group(2)}")
    #     reward = [100.0]
    
    # return reward, environment_info