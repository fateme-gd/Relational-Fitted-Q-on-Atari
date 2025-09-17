import numpy as np
import torch as th


def reward_function(self) -> float:
    for obj in self.objects:
        if 'player' in str(obj).lower():
            player = obj
            break

    for obj in self.objects:
        # print("Object: ", obj)
        if 'score' in str(obj).lower():
            score = obj
            break
            # return score.value

    # reward = -0.1  # staying on the ladder or not being climbing

    # # print(f"player.y:  {player.y}  player.prev_y: {player.prev_y}  is climbing {player.climbing}")

    # if player.climbing:
    #     if player.y < player.prev_y:    # climbing up
    #         reward = 10
    #     if player.y > player.prev_y:    # climbing down
    #         reward = -20
    
    # if player.crashed:
    #     reward = -50

    """    Reward function when there is enemy and we do not want reward engineering"""
    # reward = self.org_reward

    # if player.crashed:
    #     reward = -200
    """Comment the above section if you want to use reward engineering"""
    return self.org_reward