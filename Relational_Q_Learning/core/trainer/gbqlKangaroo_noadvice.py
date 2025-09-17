from collections import OrderedDict, defaultdict
import os
from Relational_Q_Learning.core.learning_rate_strategy.decay import LinearDecay
from Relational_Q_Learning.core.learning_rate_strategy.learning_rate_strategy import LearningRateStrategy
from Relational_Q_Learning.core.trainer.advice import AdviceHandler, AdviceRule, compute_entropy_for_qtable, compute_uncertainty_stats, seek_advice
from ins.envs.kangaroo.reward_eng import clear_environment_info, reward_engineering
from Relational_Q_Learning.srlearn import Background, Database
from ..srl import RDNRegressor
import numpy as np
from ..data_management import ReplayBuffer
from .trainer import Trainer
from ..util.logging import logger
from ..util.eval_util import create_stats_ordered_dict
import random
import gtimer as gt
from ..exploration_strategy import EpsilonGreedy
import torch
from ..util.save_model import save_image
from torch.utils.tensorboard import SummaryWriter
from typing import List

ACTION_LIST = ["noop", "fire", "up", "right", "left", "down"]


def apply_advice_to_q_values(
    idx,
    q_value_array: List[float],
    state_predicates: List[str],
    advice_handler: AdviceHandler,
    epsilon: float = 10
) -> List[float]:
    """
    If advice applies, boost the Q-value of the preferred action.
    """
    # idx = None
    q_values = q_value_array.copy()
    preferred_action = advice_handler.get_advised_action(state_predicates)
    pref_q_value = None
    
    if preferred_action: # ToDo: and preferred_action in ACTION_LIST:
        # idx = ACTION_LIST.index(preferred_action)
        if preferred_action == idx:
            return q_values[idx], idx
        max_q = max(q_values)
        if q_values[preferred_action] <= max_q:
            # print(f"[Advice] Boosting Q-value of '{preferred_action}' to {max_q + epsilon}")
            q_values[preferred_action] = max_q + epsilon
            pref_q_value = q_values[preferred_action]
    return pref_q_value, preferred_action


def get_key_from_value(d, value):
    for key, val in d.items():
        if val == value:
            return key
    return None

def get_diagnostics(paths, **kwargs):
        successes = [p['is_success'] for p in paths]
        rewards = [p['episode_reward'] for p in paths]
        # percent_solved = [p['percent_solved'] for p in paths]
        average_reward = np.mean(rewards)
        reward_max = np.max(rewards)
        success_rate = sum(successes) / len(successes)
        lengths = [p['episode_length'] for p in paths]
        length_rate = sum(lengths) / len(lengths)
        return {'Success Rate': success_rate,
                'Episode length Mean': length_rate,
                'Episode length Min': min(lengths),
                'Episode counts': len(paths),
                # 'Percent Solved Mean': np.mean(percent_solved),
                # 'Percent Solved Max': np.max(percent_solved),
                'Total Reward Mean': average_reward,
                'Total Reward Max': reward_max}


class GBQL(Trainer):
    def __init__(self, n_iter=100, n_trees=5, batch_size=10,
                 train_env=None, bk=None, max_trajectory_length=4000,
                 replay_sampling_rate=0.10, test_env=None, agent = None,
                 max_buffer_size=100000, target_predicate="q_value",
                 learning_rate=0.9, ad_coef=0.9, use_advice = False,discount_factor=0.99,
                 n_evaluation_trajectories=10, n_burn_in_traj=0,
                 additional_facts=None, goal_q_value=200,  
                 exploration_strategy=EpsilonGreedy(), device = None, learning_rate_strategy=LinearDecay(),
                 buffer=ReplayBuffer, test_gap=10):
        self.n_iterations = n_iter
        self.n_trees = n_trees
        self.batch_size = batch_size
        self.env = train_env
        self.target = target_predicate
        self.n_estimators = []
        self.test_env = test_env
        if test_env is None:
            self.test_env = train_env
        self.agent = agent           #added agent here
        self.max_traj_len = max_trajectory_length
        self.buffer = buffer(max_size=max_buffer_size)  #a queue of size 1000 (max_buffer_size)
        self.replay_sampling_rate = replay_sampling_rate
        self.learning_rate = learning_rate
        self.ad_coef = ad_coef
        self.use_advice = use_advice
        self.discount_factor = discount_factor
        self.n_eval_traj = n_evaluation_trajectories
        self.burn_in_traj = n_burn_in_traj
        self.bk = bk
        if bk is None:
            self.bk = Background()
        self.additional_facts = additional_facts
        self.goal_qvalue = goal_q_value
        self.exploration_strategy = exploration_strategy
        self.learning_rate_strategy = learning_rate_strategy
        self.test_gap = test_gap
        self.device = device

        self.advice = []
        self.handler = None
        self.adviceBudget = 3
        # if self.use_advice:
        #     self.advice = [
        #             # AdviceRule(["closeByMonkey(P, M)"], preferred_action=1),
        #             # AdviceRule(["leftOfLadder(P, L)"], preferred_action=3),
        #             # AdviceRule(["rightOfLadder(P, L)"], preferred_action=4),
        #             AdviceRule(["onLadder(P, L)"], preferred_action=2),
        #             ]     
        #     self.handler = AdviceHandler(self.advice)
        
        self.all_experiment_avg_step = []
        self.all_experiment_avg_reward = []
        self.all_experiment_avg_bellman_error = []
        self.all_experiment_test_avg_reward = []
        self.all_experiment_test_avg_length = []

        if learning_rate_strategy is not None:
            self.learning_rate = self.learning_rate_strategy.alpha

    def fit_q(self, train, target, path=None, save=False):
        """Learn a relational Q Function using RDN Boost"""
        bk = self.bk

        reg = RDNRegressor(background=bk, target=target, n_estimators=self.n_trees) 
        if self.additional_facts is not None:
            train.facts += self.additional_facts
        reg.fit(train, path, preserve_data=save)

        return reg
    
    
    def reset(self, env):
        """Reset the environment and clear the background knowledge"""
        done = True
        goal_reached = True
        current_state = []

        while done or goal_reached or len(current_state) == 0: 
            next_logic_obs, _ = env.reset()
            action = random.choice([0,1,2,3,4,5])
            (next_logic_obs, img ), reward, done, _ , _ = env.step(action)
            done = done[0]
            reward = torch.tensor(reward).to(self.device).view(-1)
            reward = reward.cpu().numpy()
            if reward < -0.5:
                done = True
            self.agent.compute_init_v(next_logic_obs)  # we need to pass the state to logic actor to compute the init v
            current_state, goal_reached = self.agent.print_valuations_input(self.agent.V_0, min_value=0.7)
        
        return current_state
    

    def polish_states(self, current_state, state_id):
        # c_state = []
        # for state in current_state:
        #     if 'closeByMonkey' in state:
        #         c_state.append(state)
        # if len(c_state) > 0:
        #     current_state = c_state
        modified_states = []
        for state in current_state:
            predicate, args = state.split('(', 1)
            args = args.rstrip(').')
            modified_state = f"{predicate}({args},s{state_id})."
            modified_states.append(modified_state)
        
        return current_state, modified_states

    def generate_batch(self, train_batch, batch_size=10, q_function=None, q_table=None):
        """Generate a training batch"""
        # q_table = {}

        state_id = 0
        goal_reached = True
        current_state = []

        bellman_errors = []
        traj_reward = []

        batches = batch_size 
        max_traj_len = self.max_traj_len 

        state_stats = defaultdict(lambda: {'remained': 0, 'departed': 0, 'arrived': 0, 'crashed': 0})

        for i in range(batches):
            print("Generating batch: ", i)

            """keep reseting the env if it starts at the end or the logic representation is empty"""
            current_state = self.reset(self.env)

            print(f"Current state: {state_id}{current_state}")
            # save_image(img, f"img/state_{state_id}.png")
            done = False
            traj_len = 0
            ret = 0 

            while not done:
                current_state, modified_states = self.polish_states(current_state, state_id= state_id)
                     
                action, q_value, advice_action, advice_qvalue,_ = self.get_action(current_state, q_function, use_advice=self.use_advice, q_table=q_table)   
                action_key = get_key_from_value(self.env.pred2action, action)
                action_key = "{ACTION}({player},{state_id})".format(ACTION = action_key, player = "obj1", state_id=f"s{state_id}")
                # trajectory.append((current_state, action))  #cause later we need the action from trajectory to step in the env. if sth else is needed to be done with trajectory. then we have to erite a reverse action func

                (next_state, img_0 ), reward , done, _ , _ = self.env.step(action)
                reward = reward[0] if isinstance(reward, list) else reward
                done = done[0]
                   
                self.agent.compute_init_v(next_state)
                next_state, goal_reached = self.agent.print_valuations_input(self.agent.V_0, min_value=0.7)
                
                reward, crashed = reward_engineering(reward=reward, current_state = current_state, next_state=next_state)

                if next_state == current_state:
                    state_stats[tuple(current_state)]['remained'] += 1
                elif len(next_state) == 0:
                    state_stats[tuple(current_state)]['crashed'] += 1
                else:
                    state_stats[tuple(current_state)]['departed'] += 1
                    state_stats[tuple(next_state)]['arrived'] += 1

                    
                if crashed:
                    current_state = self.reset(self.env)
                    continue  #instead of break
                
                if len(next_state)==0 and reward >= 0:
                    print(f"next_state in {state_id} is empty")
                    save_image(img_0, f"img/state_{state_id}_1.png")
                    next_state = current_state 

                killed = False
                if reward < -0.5:  #when monkey killed the player
                    killed = True #done = True

                ret += reward
                
                if goal_reached:
                    current_state = self.reset(self.env) # done = True                             
                    next_state_qvalue = self.goal_qvalue
                    print(f"Reached goal, setting next state Q-value to goal Q-value: {next_state_qvalue}")
                elif killed:
                    current_state = self.reset(self.env)
                    next_state_qvalue = -1
                    # print(f"st: {state_id}, C_s:{current_state}, N_s:{next_state}, a:{action}, advised_action:{advice_action}, r:{reward} , done:{done}, goal_reached:{goal_reached}")
                    # print(f"Player killed, setting next state Q-value to -200")
                    
                else:
                    current_state = next_state
                    """Since this is for bellman error calculation, we need to get the best next state Q-value without considering advice"""
                    n_action, next_state_qvalue, n_a_action, n_a_qvalue,_ = self.get_action(next_state, q_function, best_train=True, q_table=q_table)
                    
                traj_len += 1
                if traj_len >= max_traj_len :
                    done = True 

                new_q_value = reward + (self.discount_factor * next_state_qvalue)
                q = ((1.0 - self.learning_rate) * q_value) + (self.learning_rate * new_q_value)
                
                if action == advice_action :
                    q = ((1.0 - self.learning_rate) * q_advice) + (self.learning_rate * new_q_value)
                    q = q_advice
                
                bellman_errors.append(abs(q_value - (reward + (self.discount_factor * next_state_qvalue))))
                

                train_batch.facts += modified_states   
                train_batch.pos.append(f"regressionExample({action_key},{q:.3f}).")

                state_id += 1
            traj_reward.append(ret)
            print("reward: ", ret)
        # seek_advice(train_batch.facts, train_batch.pos)
        print("State statistics:", state_stats)
        return (state_id /batch_size), np.mean(traj_reward), np.mean(bellman_errors), state_stats   #average step size, average reward, average bellman error

    def get_training_batch(self, batch_size, q_function, q_table):
        train_batch = Database()

        gt.stamp('sampled historic trajectories', unique=False)
        avg_steps, avg_rewards , avg_bellman_error, state_stat = self.generate_batch(train_batch, batch_size, q_function, q_table=q_table)
        gt.stamp('sampled new trajectories', unique=False)
        
        # self.buffer.add_all_trajectories(new_traj)
        
        gt.stamp('evaluate historic trajectories', unique=False)
        
        return train_batch, avg_steps, avg_rewards , avg_bellman_error, state_stat
    

    def get_action(self, state, q_function=None, q_table=None, env=None, best_train=False, best_test=False, print_test=None, use_advice=False):
        if env is None:
            env = self.env

        possible_actions = list(env.pred2action.values())[0:5]  # {'noop': 0, 'fire': 1,'up': 2, 'right': 3, 'left': 4, 'down': 5}
        advice_adjusted_q = prefered_action = None
        rng = random.Random()

        if q_function is None:
            q_values = [0.0] * len(possible_actions)
            action = random.choice(possible_actions)  # Use imitation learning here if available
            if use_advice:
                advice_adjusted_q, prefered_action = apply_advice_to_q_values(q_values, state, self.handler)
                if prefered_action is not None:
                    epsilon = 1 - self.ad_coef
                    if rng.random() > epsilon:
                        action = prefered_action
            return action, 0.0, prefered_action, advice_adjusted_q, q_table

        idx, q_values, best_action = self.predict(q_function, q_table=q_table, state=state, additional_facts=self.additional_facts, print_test=print_test)


        if use_advice:
            advice_adjusted_q, prefered_action = apply_advice_to_q_values(idx, q_values, state, self.handler)

        # Add noise during test time for generalization
        if best_test and not best_train and rng.random() <= 0.2:
            idx = rng.randrange(len(possible_actions))

        # Training mode with exploration + advice injection
        if not best_train and not best_test:
            idx = self.exploration_strategy.get_action_idx(idx, len(possible_actions))
            if prefered_action is not None:
                epsilon = 1 - self.ad_coef
                if rng.random() > epsilon:
                    idx = prefered_action

        return possible_actions[idx], q_values[idx], prefered_action, advice_adjusted_q, q_table
    
    def fill_q_table(self, q_table, updated_q, state_stat):
        """
        Fill the Q-table with the updated Q-values from the regressor.
        """
        q_table = {}
        for state in state_stat.keys():
            # temp_state = []
            # for pred in state:
            #     temp_state.append(pred+"(obj1,obj0).")
            _ , q_values, _= self.predict(state = state, q_function= updated_q, q_table= q_table)
            if state not in q_table:
                q_table[tuple(state)] = q_values
        return q_table


    def train(self):
        """Fitted Q Learning"""
        current_q = None
        q_table = {}
        # current_q = RDNRegressor()
        # current_q.from_json(f"out/test/gbql-stack-2025_07_25_10_29_39--seed0--exp-Id-0/")#itr_0.json"
        
        writer_base_dir = f"{logger.get_snapshot_dir()}/tensorboard"
        os.makedirs(writer_base_dir, exist_ok=True)
        writer = SummaryWriter(writer_base_dir)
        
        logger.log("started fitted Q training")

        for i in gt.timed_for(range(self.n_iterations), save_itrs=True):
            logger.log(f"Iteration {i} started")
            logger.log(f"Iteration {i} getting training batch")
            
            if q_table != {}:
                print("q_table: ", q_table)
                q_table_entropy = compute_entropy_for_qtable(q_table, temperature=1.0)
                print("Q-table entropy:", q_table_entropy)
                max_state = max(q_table_entropy, key=q_table_entropy.get)
                print("max_state:", max_state)
                
                # if self.adviceBudget > 0:
                    
                #     if 'onLadder' in max_state:
                #         self.advice.append(AdviceRule(["onLadder(P, L)"], preferred_action=2))
                #         self.adviceBudget -= 1
                #     elif 'closeByMonkey' in max_state:
                #         self.advice.append(AdviceRule(["closeByMonkey(P, M)"], preferred_action=1))
                #         self.adviceBudget -= 1
                #     elif max_state == 'rightOfLadder':
                #         self.advice.append(AdviceRule(["rightOfLadder(P, L)"], preferred_action=4))
                #         self.adviceBudget -= 1
                #     elif max_state == 'leftOfLadder':
                #         self.advice.append(AdviceRule(["leftOfLadder(P, L)"], preferred_action=3))
                #         self.adviceBudget -= 1
                #     else:
                #         preferred_action=input(f"Enter desired action for state '{max_state}' (remaining advice budget: {self.adviceBudget}): ")
                #         # preferred_body=input(f"Enter desired body for state '{max_state}', hint: 'onLadder(P, L).','closeByMonkey(P, M).'")
                #         print("preferred_action:", preferred_action)
                #         # print("preferred_body:", preferred_body)

                #         self.advice.append(AdviceRule(list(max_state), preferred_action=int(preferred_action)))
                #         self.adviceBudget -= 1

                # self.handler = AdviceHandler(self.advice)
                # self.use_advice = True
                # for adv in self.advice:
                #     print(f"Advice: {adv.rule_bodies}, Preferred Action: {adv.preferred_action}")

            train_batch, avg_steps, avg_rewards , avg_bellman_error, state_stat = self.get_training_batch(self.batch_size, current_q, q_table=q_table)

            writer.add_scalar("charts/train_avg_steps", avg_steps, i)
            writer.add_scalar("charts/train_avg_reward", avg_rewards, i)
            writer.add_scalar("charts/train_avg_bellman_error", avg_bellman_error, i)
           

            gt.stamp("training batch", unique=False)
            logger.log(f"Iteration {i} fitting q function")

            updated_q = self.fit_q(train_batch, target="noop,fire,up,right,left",path=f"{logger.get_snapshot_dir()}/fitted-q/itr{i}", save=True) #returns a regressor

            # q_table = {}
            q_table = self.fill_q_table(q_table, updated_q, state_stat)

            
            gt.stamp("bsrl learning", unique=False)
        
            self.n_estimators.append(updated_q) 
            self.exploration_strategy.end_epoch()
            
            if self.learning_rate_strategy is not None:
                    self.learning_rate = self.learning_rate_strategy.end_epoch()
                    self.ad_coef = self.learning_rate
            
            # current_q = RDNRegressor()
            # current_q.from_json(f"out/test/gbql-stack-2025_07_25_10_29_39--seed0--exp-Id-0/itr_{i}.json")
            # updated_q = current_q   #remove this after test
            
            if i >= 0: 
                logger.log(f"Iteration {i} evaluating")
                paths, q_table = self.evaluate(self.n_eval_traj, updated_q, q_table=q_table)

                gt.stamp("bsrl evaluation", unique=False)
                training_stats = None
                self._log_stat(updated_q, training_stats, paths, i)
                logger.record_dict(self.exploration_strategy.stats(), prefix='exploration/')
                logger.dump_tabular()

                test_avg_total_reward = 0.0
                test_avg_length = 0.0
                for k in paths:
                    test_avg_total_reward += k['episode_reward']
                    test_avg_length += k['episode_length']

                writer.add_scalar("charts/test_episodic_return", test_avg_total_reward/len(paths), i)
                writer.add_scalar("charts/test_episode_length", test_avg_length/len(paths), i)

                self.all_experiment_avg_step.append(avg_steps)
                self.all_experiment_avg_reward.append(avg_rewards)
                self.all_experiment_avg_bellman_error.append(avg_bellman_error)
                self.all_experiment_test_avg_reward.append(test_avg_total_reward/len(paths))
                self.all_experiment_test_avg_length.append(test_avg_length/len(paths))
                
            current_q = updated_q

            logger.log(f"Iteration {i} ended")
        print("All experiment average step: ", self.all_experiment_avg_step)
        print("All experiment average reward: ", self.all_experiment_avg_reward)
        print("All experiment average bellman error: ", self.all_experiment_avg_bellman_error)
        print("All experiment test average reward: ", self.all_experiment_test_avg_reward)
        print("All experiment test average length: ", self.all_experiment_test_avg_length)  

        return current_q, self.all_experiment_avg_bellman_error, self.all_experiment_avg_reward, \
               self.all_experiment_avg_step, self.all_experiment_test_avg_reward, self.all_experiment_test_avg_length

    def _log_stat(self, q_function, training_stats, paths, itr):
        logger.save_itr_params(itr, q_function)
        # logger.record_dict(training_stats, prefix='training/')
        # buffer_stats = self.buffer.get_diagnostics()
        # logger.record_dict(buffer_stats, prefix='buffer/')
        evaluation_stats = get_diagnostics(paths)    
        logger.record_dict(evaluation_stats, prefix='evaluation/')
        logger.save_eval_data(paths, itr=itr)
        logger.record_tabular('iteration', itr)
        times_itrs = gt.get_times().stamps.itrs
        times = OrderedDict()
        epoch_time = 0
        for key in sorted(times_itrs):
            time = times_itrs[key][-1]
            epoch_time += time
            times['{} (s)'.format(key)] = time
        times['iteration (s)'] = epoch_time
        times['total (s)'] = gt.get_times().total
        logger.record_dict(times, prefix=f'time/')


    def evaluate(self, eval_batch_size, q_function, q_table=None):
        """Evaluation in Test env """
        paths = []
        total_reward = 0
        current_test_state = []
        goal_reached = True

        for i in range(eval_batch_size):
            print("Evaluating: ", i+1)

            total_reward = 0.0
            path = dict(states=[], episode_reward=0, is_success=0, episode_length=0, percent_solved=0.0)
            
            current_test_state = self.reset(self.test_env)
            path['states'].append(current_test_state)

            done = False
            traj_len = 0
            state_test_id = 0
            while not done:
                action, _, _, _, q_table = self.get_action(current_test_state, q_function, q_table= q_table , env=self.test_env, best_test=True, print_test=False) 
                action_key = get_key_from_value(self.env.pred2action, action)
                action_key = "{ACTION}".format(ACTION=action_key)
                
                (next_state, img ), reward, done, _ , _ = self.test_env.step(action)
                self.agent.compute_init_v(next_state)
                next_symbolic, goal_reached = self.agent.print_valuations_input(self.agent.V_0, min_value=0.7)
                
                done = done[0]
                r = reward[0] if isinstance(reward, list) else reward
                r, crashed = reward_engineering(reward=r, current_state = current_test_state, next_state=next_symbolic)

                total_reward += r
                traj_len += 1
                state_test_id += 1

                if goal_reached:
                    path['is_success'] += 1

                if traj_len >= self.max_traj_len: 
                    path['states'].append(next_symbolic) #last state
                    path['episode_length'] = traj_len
                    path['episode_reward'] = total_reward
                    done = True
                
                if r < -0.5 or crashed or len(next_symbolic)==0 :
                    current_test_state = self.reset(self.test_env)
                    continue 

                current_test_state = next_symbolic

            paths.append(path)
            print(path)
        return paths, q_table


    def predict(self, q_function, q_table=None, state=None, additional_facts=None, print_test = None):   
        
        state_id = "s1"
        all_actions = list(self.env.pred2action.values())[0:5]

        test = Database()
        modified_test_states = self.polish_states(state, state_id=state_id)[1]  # polish the states to have the correct format
        test.facts = modified_test_states
        # print("modified_test_states: ", modified_test_states)
        
        q_test_values = 0.0        # it use this as true value. In block it is always zero
        for action in all_actions:
            action_key = get_key_from_value(self.env.pred2action, action)
            action_key = "{ACTION}({player},{state_id})".format(ACTION = action_key, player = "obj1", state_id=f"s{state_id}")
            test.pos.append(f"regressionExample({action_key},{q_test_values:.3f}).")
        
        if additional_facts is not None:
            test.facts += additional_facts
        
        if tuple(state) not in q_table:
            #To Do: if it comes from evaluation, we calculate the uncertainty as well and add to uncertainty table
            q_values = q_function.predict(test)     #rql prediction
            q_table[tuple(state)] = q_values
        
        # print(q_table)
        q_values = q_table[tuple(state)]

        if print_test:
            print("q_values: ", q_values)  #up,right,left,down

        where_max = np.where(q_values == np.max(q_values))[0]
        if len(where_max) == 1:
            idx = where_max[0]
        else:
            idx = np.random.choice(where_max)
        return idx, q_values, test.pos[idx]


class RRT(GBQL):
    """The RRT code is same as GBQL but with only 1 tree"""

    def __init__(self, n_iter=1, batch_size=10, train_env=None, bk=None, agent=None, max_trajectory_length=50,
                 replay_sampling_rate=0.10, test_env=None, max_buffer_size=1000, target_predicate="q_value",
                 learning_rate=0.9, ad_coef=0.9, use_advice = False, discount_factor=0.99, n_evaluation_trajectories=10,
                 n_burn_in_traj=0, additional_facts=None, goal_q_value=100, exploration_strategy=EpsilonGreedy(), device=None,
                 learning_rate_strategy=None, buffer=ReplayBuffer, test_gap=10):
        super().__init__(n_iter=n_iter, n_trees=1, batch_size=batch_size, train_env=train_env, bk=bk,
                         max_trajectory_length=max_trajectory_length, replay_sampling_rate=replay_sampling_rate,
                         test_env=test_env, agent=agent, max_buffer_size=max_buffer_size, target_predicate=target_predicate,
                         learning_rate=learning_rate, ad_coef=ad_coef, use_advice = use_advice, discount_factor=discount_factor,
                         n_evaluation_trajectories=n_evaluation_trajectories, n_burn_in_traj=n_burn_in_traj,
                         additional_facts=additional_facts, goal_q_value=goal_q_value,
                         exploration_strategy=exploration_strategy, device = device, learning_rate_strategy=learning_rate_strategy,
                         buffer=buffer, test_gap=test_gap)


