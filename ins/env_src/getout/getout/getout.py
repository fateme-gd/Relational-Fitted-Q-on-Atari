import pathlib

from .actions import GetoutActions
from .camera import Camera
from .hud import HUD
from .level import Level
from .trackingCamera import TrackingCamera
from .resource_loader import ResourceLoader
from .player_v1 import Player
from gymnasium.spaces import Discrete

import gymnasium as gym


# class Getout():

#     def __init__(self, render=True, resource_path=None, start_on_first_action=False, width=50):
#         # self.unwrapped = self
#         self.action_space = Discrete(5)
#         self.observation_space = Discrete(8)  # TODO
#         self.zoom = 42 - width//2

#         if resource_path is None:
#             resource_path = pathlib.Path(__file__).parent.joinpath('../assets/kenney/')
#         self.resource_loader = ResourceLoader(path=resource_path, sprite_size=self.zoom,
#                                               no_loading=not render) if render else None

#         """ change here to choose  mode """

#         self.score = 0.0
#         self.level = Level(width, 16)
#         self.player = Player(self.level, 2, 2, self.resource_loader)
#         self.level.entities.append(self.player)
#         self.width = width

#         # self.camera = TrackingCamera(900, 600, self.player, zoom=self.zoom) if render else None
#         self.camera = Camera(900, 600, x=-10, y=-50, zoom=self.zoom) if render else None
#         self.show_step_counter = False
#         self.hud = HUD(self, self.resource_loader)
#         # start stepping the game only after the first action.
#         # this is useful when recording human run-throughs to avoid
#         self.start_on_first_action = start_on_first_action
#         self.has_started = not start_on_first_action

#         self.step_counter = 0

#     def clear(self):
#         raise NotImplementedError()

#     def step(self, action):
#         if self.level.terminated:
#             return 0

#         # start stepping the game only after the first action.
#         if self.start_on_first_action and not self.has_started:
#             if isinstance(action, int):
#                 la = 0
#             else:
#                 la = len(action)
#             if la == 0 or (la == 1 and action[0] == GetoutActions.NOOP):
#                 return None
#             else:
#                 self.has_started = True

#         self.step_counter += 1

#         self.player.set_action(action)
#         self.level.step()

#         self.render()

#         reward = self.level.get_reward()
#         # if self.score + reward <= 0 and not self.level.terminated:
#         #     # terminate if the score drops below zero
#         #     self.level.terminate(True)
#         #     reward = self.level.get_reward()  # losing changes the reward
#         self.score += reward

#         return reward

#     def render(self):
#         if self.camera is not None:
#             self.camera.start_render()
#             self.level.render(self.camera, self.step_counter)
#             self.camera.end_render()

#             self.hud.render(self.camera, step=self.step_counter if self.show_step_counter else None)

#     def get_score(self):
#         return self.score

    # def reset(self):
    #     import ipdb; ipdb.set_trace()
    #     self.step_counter = 0
    #     return [0 for _ in range(8)]






class Getout(gym.Env):

    def __init__(self, render=True, resource_path=None, start_on_first_action=False, width=50):
        # self.unwrapped = self
        self.action_space = Discrete(5)
        self.observation_space = Discrete(8)  # TODO
        self.zoom = 42 - width//2

        if resource_path is None:
            resource_path = pathlib.Path(__file__).parent.joinpath('../assets/kenney/')
        self.resource_loader = ResourceLoader(path=resource_path, sprite_size=self.zoom,
                                              no_loading=not render) if render else None

        """ change here to choose  mode """

        self.score = 0.0
        self.level = Level(width, 16)
        self.player = Player(self.level, 2, 2, self.resource_loader)
        self.level.entities.append(self.player)
        self.width = width

        # self.camera = TrackingCamera(900, 600, self.player, zoom=self.zoom) if render else None
        self.camera = Camera(900, 600, x=-10, y=-50, zoom=self.zoom) if render else None
        self.show_step_counter = False
        self.hud = HUD(self, self.resource_loader)
        # start stepping the game only after the first action.
        # this is useful when recording human run-throughs to avoid
        self.start_on_first_action = start_on_first_action
        self.has_started = not start_on_first_action

        self.step_counter = 0

        self.env_name = 'getout'

    def clear(self):
        raise NotImplementedError()

    def step(self, action):
        if self.level.terminated:
            return 0

        # start stepping the game only after the first action.
        if self.start_on_first_action and not self.has_started:
            if isinstance(action, int):
                la = 0
            else:
                la = len(action)
            if la == 0 or (la == 1 and action[0] == GetoutActions.NOOP):
                return None
            else:
                self.has_started = True

        self.step_counter += 1

        self.player.set_action(action)
        self.level.step()

        self.render()

        reward = self.level.get_reward()
        # if self.score + reward <= 0 and not self.level.terminated:
        #     # terminate if the score drops below zero
        #     self.level.terminate(True)
        #     reward = self.level.get_reward()  # losing changes the reward
        self.score += reward

        return reward

    def render(self):
        if self.camera is not None:
            self.camera.start_render()
            self.level.render(self.camera, self.step_counter)
            self.camera.end_render()

            self.hud.render(self.camera, step=self.step_counter if self.show_step_counter else None)

    def get_score(self):
        return self.score
    
    def reset(self):
        # import ipdb; ipdb.set_trace()
        self.step_counter = 0
        return [0 for _ in range(8)]

    # def reset(self):
    #     self.level = Level(self.width, 16)
    #     self.player = Player(self.level, 2, 2, self.resource_loader)
    #     self.level.entities = [self.player]
    #     self.score = 0.0
    #     self.step_counter = 0
    #     self.has_started = not self.start_on_first_action

    #     return [0 for _ in range(8)] 
