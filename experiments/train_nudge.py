import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
import sys

# Add the correct blendrl directory to sys.path
visual_fitted_q_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
blendrl_path = os.path.join(visual_fitted_q_path, 'blendrl')
nudge_path = os.path.join(visual_fitted_q_path, 'nudge')
relational_q_learning_path = os.path.join(visual_fitted_q_path, 'Relational_Q_Learning')

# added
from blendrl.agents.blender_agent import BlenderActorCritic
from blendrl.env_vectorized import VectorizedNudgeBaseEnv
from blendrl.nudge.utils import save_hyperparams
import os
import sys
import time
from pathlib import Path

import pickle
import random
import numpy as np
from rtpt import RTPT

from blendrl.nudge.utils import load_model_train

############  Fitted Q Imports ######
from Relational_Q_Learning.core.trainer import RRT, GBQL
from Relational_Q_Learning.core.exploration_strategy import EpsilonGreedyWithExponentialDecay
from Relational_Q_Learning.core.util.launcher_util import setup_logger
import gtimer as gt
from Relational_Q_Learning.srlearn import Background
######################################

# Log in to your W&B account
import wandb
OUT_PATH = Path("out/")
IN_PATH = Path("in/")

torch.set_num_threads(5)

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 0
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "blendeRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Seaquest-v4"
    """the id of the environment"""
    total_timesteps: int = 60000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    
    # added
    env_name: str = "kangaroo"
    """the name of the environment"""
    algorithm: str = "blender"
    """the algorithm used in the agent"""
    blender_mode: str = "logic"
    """the mode for the blend (logic or neural)"""
    blend_function: str = "softmax"
    """the function to blend the neural and logic agents: softmax or gumbel_softmax"""
    actor_mode: str = "hybrid"
    """the mode for the agent"""
    rules: str = "default"
    """the ruleset used in the agent"""
    save_steps: int = 5000000
    """the number of steps to save models"""
    pretrained: bool = False
    """to use pretrained neural agent"""
    joint_training: bool = False
    """jointly train neural actor and logic actor and blender"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer (neural)"""
    logic_learning_rate: float = 2.5e-4
    """the learning rate of the optimizer (logic)"""
    blender_learning_rate: float = 2.5e-4
    """the learning rate of the optimizer (blender)"""
    blend_ent_coef: float = 0.01
    """coefficient of the blend entropy"""
    recover: bool = False
    """recover the training from the last checkpoint"""
    reasoner: str = "nsfr"
    """the reasoner used in the agent; nsfr or neumann"""


def main():
        
    args = tyro.cli(Args)
    rtpt = RTPT(name_initials='HS', experiment_name='BlendeRL', max_iterations=int(args.total_timesteps / args.save_steps))
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    model_description = "{}_blender_{}".format(args.blend_function, args.blender_mode)
    learning_description = f"lr_{args.learning_rate}_llr_{args.logic_learning_rate}_blr_{args.blender_learning_rate}_gamma_{args.gamma}_bentcoef_{args.blend_ent_coef}_numenvs_{args.num_envs}_steps_{args.num_steps}_"
    run_name = f"{args.env_name}_{model_description}_{learning_description}_{args.seed}"
    if args.track:
        wandb.init(
            project=args.wandb_project_name + "_" + args.env_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name = run_name,
            monitor_gym=True,
            save_code=True,
        )

    # for logging and model saving
    experiment_dir = OUT_PATH / "runs" / run_name # / now.strftime("%y-%m-%d-%H-%M")
    checkpoint_dir = experiment_dir / "checkpoints"
    writer_base_dir = OUT_PATH / "tensorboard" # Path("tensorboard")
    writer_dir = writer_base_dir / run_name
    image_dir = experiment_dir / "images"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(writer_dir, exist_ok=True)
    
    writer = SummaryWriter(writer_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

                    
    ######################   FITTED-Q    ##########################################
    variant = {
        'trainer': 'gbql',
        
        'bk_kwargs': {'max_tree_depth': 2,
                    'node_size': 2,
                    'ok_if_unknown':  ['onLadder/3', 'rightLadder/3', 'leftLadder/3','sameLevelChild/3'],
                    },
        'trainer_kwargs': {'n_iter': 100,
                        # 'max_buffer_size': 500,
                        # 'target_predicate': 'move', 
                        'learning_rate':0.1,
                        'test_gap':10 },
        'exploration_strategy': EpsilonGreedyWithExponentialDecay,
        'modes':
                [
                # "close_by_fruit(+state,#oplayer, #ofruit).",
                # "close_by_bell(+state,#oplayer, #obell).",
                # "close_by_monkey(+state,+oplayer, +omonkey).",
                # "close_by_throwncoconut(+state,+oplayer, +othrowncoconut).",
                # "close_by_fallingcoconut(+state,+oplayer, +ofallingcoconut).",
                # "nothing_around(+state).",
                
                # "on_pl_ladder(+state,-oladder, +oplatform).",
                # "on_pl_player(+state,-oplayer, -oplatform).",
                
                "onLadder(+player,-ladder,+state).",
                "rightLadder(+player,-ladder,+state).",
                "leftLadder(+player,-ladder,+state).",
                "sameLevelChild(+player,-child,+state).",
                
                "up(+player,+state).",
                "down(+player,+state).",
                "left(+player,+state).",
                "right(+player,+state)."]
                }


    n_iter = 1
    for i in range(n_iter):
        variant['experiment_no'] = i
        setup_logger(f"{variant['trainer']}-stack", variant=variant, snapshot_mode="all", exp_id=i)
        
        train_env = VectorizedNudgeBaseEnv.from_name(args.env_name, n_envs=args.num_envs, mode=args.algorithm, seed=args.seed)#$, **env_kwargs)
        agent = BlenderActorCritic(train_env, args.rules, args.actor_mode, args.blender_mode, args.blend_function, args.reasoner, device)
        bk = Background(modes=variant['modes'], **variant['bk_kwargs'])
        test_env = VectorizedNudgeBaseEnv.from_name(args.env_name, n_envs=args.num_envs, mode=args.algorithm, seed=args.seed)#$, **env_kwargs)

        RRT_Trainer = GBQL(train_env=train_env, bk=bk, test_env=test_env, agent=agent, exploration_strategy=variant['exploration_strategy'](), device = device, 
                        **variant['trainer_kwargs'])
        
        fitted_q = RRT_Trainer.train()
        gt.reset_root()

    ################################################################################################################
     
   
if __name__ == "__main__":
    main()