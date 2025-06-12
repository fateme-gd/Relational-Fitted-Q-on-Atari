import numpy as np
import torch as th


def reward_function(self) -> float:
    for obj in self.objects:
        if 'player' in str(obj).lower():
            player = obj
            break

    # got reawrd and previous step was on the top platform -> reached the child
    # if game_reward == 1.0 and player.prev_y == 4:
    #    reward = 10.0
    # x = 129
    # if player.y == 4:
        # reward = 0.2
    # BUG ↓ with multi envs, rewards collected repeatedlydd
    # print("player.y: ", player.y)
    # print("player.prev_y: ", player.prev_y)
    # print("self.org_reward: ", self.org_reward)
    # print(self.objects)

    # if 28 > player.y and 28 < player.prev_y and player.prev_y < 76:
    #     reward = 100.0 #changed from 20 to 100 for the purpose of R Fitted Q
    # elif self.org_reward > 1.0 and self.org_reward < 5:  # from 1 -> 2
    #     reward = 20.0
    
    # ################Added for Fitted Q#############
    # elif (28 < player.y and player.y < 76 and 76 <= player.prev_y and player.prev_y <= 124) or (76 < player.y and player.y < 124 and 124 <= player.prev_y and 172 >= player.prev_y) or (124 < player.y and player.y < 172 and 172 <= player.prev_y ):
    #     reward = 10.0
    ################Added for Fitted Q############
    
    # else:
    #     reward = -player.y
    
    reward = -0.1  # staying on the ladder or not being climbing

    # print(f"player.y:  {player.y}  player.prev_y: {player.prev_y}  is climbing {player.climbing}")

    if player.climbing:
        if player.y < player.prev_y:    # climbing up
            reward = 10
        if player.y > player.prev_y:    # climbing down
            reward = -20
    
    # if player.crashed:
    #     reward = -200
                      

    return reward