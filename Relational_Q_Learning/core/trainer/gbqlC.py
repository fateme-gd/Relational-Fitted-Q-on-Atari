from collections import OrderedDict
from ins.envs.kangaroo.reward_eng import clear_environment_info, reward_engineering
from srlearn import Background, Database
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
from ..util.save_model import log_trajectory, save_image



def get_key_from_value(d, value):
    for key, val in d.items():
        if val == value:
            return key
    return None

def get_diagnostics(paths, **kwargs):
        successes = [p['is_success'] for p in paths]
        rewards = [p['episode_reward'] for p in paths]
        percent_solved = [p['percent_solved'] for p in paths]
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
                 train_env=None, bk=None, max_trajectory_length=5000,
                 replay_sampling_rate=0.10, test_env=None, agent = None,
                 max_buffer_size=100000, target_predicate="q_value",
                 learning_rate=0.9, discount_factor=0.99,
                 n_evaluation_trajectories=2, n_burn_in_traj=0,
                 additional_facts=None, goal_q_value=200,  # ToDo check if ok
                 exploration_strategy=EpsilonGreedy(), device = None, learning_rate_strategy=None,
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
        if learning_rate_strategy is not None:
            self.learning_rate = self.learning_rate_strategy.alpha

    def fit_q(self, train, target, path=None, save=False):
        """Learn a relational Q Function using RDN Boost"""
        bk = self.bk
        # target = q_value
        reg = RDNRegressor(background=bk, target=target, n_estimators=self.n_trees) 
        if self.additional_facts is not None:
            train.facts += self.additional_facts
        reg.fit(train, path, preserve_data=save)
        # I guess the rest is not needed, cause in RL we dont have + or negatiive examples.
        # load cache
        
        # states = self.env.get_state_space()                     #env - in block env this method is not implemented and returns none #FIXME: This indeed needs a further check as this is the main part to fill the cache in reg
        states = None                     #env - in block env this method is not implemented and returns none #FIXME: check if this is correct
        if states is not None:
            test = Database()     # a class of pos, neg and facts 
            for i, state in enumerate(states):
                self.agent.logic_actor.compute_init_v(state)
                test_states, goal_reached =  self.agent.logic_actor.print_valuations_input(self.agent.logic_actor.V_0, min_value=0.7)      #env
                test.facts += test_states
                test.pos += list(self.env.pred2action.values())   
                # f"regressionExample({action_key},{q_value:.3f})."
            if self.additional_facts is not None:
                test.facts += self.additional_facts
            q_values = reg.predict(test)
            q_arr = q_values.reshape((len(states),int(q_values.size/len(states))))
            reg.set_cache(dict(zip(states,q_arr)))

        return reg

    def generate_batch(self, train_batch, batch_size=10, q_function=None):
        """Generate a training batch"""
        new_batch = []
        state_id = 0
        traj_lens = []
        bellman_error = []
        goal_reached = True
        current_state = []

        for i in range(batch_size):
            print("Generating batch: ", i)
            trajectory = []
            clear_environment_info()

            while goal_reached or len(current_state) == 0: 
                done = True
                next_logic_obs, img2 = self.env.reset()
                action = random.choice([2,3,4,5])
                (next_logic_obs, _ ), _, _, _ , _ = self.env.step(action)
                self.agent.logic_actor.compute_init_v(next_logic_obs)  # we need to pass the state to logic actor to compute the init v
                current_state, goal_reached = self.agent.logic_actor.print_valuations_input(self.agent.logic_actor.V_0, min_value=0.7)
            print(f"Current state: {current_state}")
        
            done = False
            traj_len = 0
            while not done:
                action, q_value = self.get_action(current_state, q_function)   #current state is the symbolic state   g  
                action_key = get_key_from_value(self.env.pred2action, action)
                action_key = "{ACTION}({player},{state_id})".format(ACTION = action_key, player = "obj1", state_id=f"s{state_id}")
                # trajectory.append((current_state, action))  #cause later we need the action from trajectory to step in the env. if sth else is needed to be done with trajectory. then we have to erite a reverse action func
                
                (next_state, img ), reward, done, _ , _ = self.env.step(action)
                self.agent.logic_actor.compute_init_v(next_state)
                next_state, goal_reached = self.agent.logic_actor.print_valuations_input(self.agent.logic_actor.V_0, min_value=0.7)

                reward = reward_engineering(str(current_state), str(next_state))
                
                reward = torch.tensor(reward).to(self.device).view(-1)
                reward = reward.cpu().numpy()
                if len(next_state)==0:
                    reward -= 1
                    next_state = current_state 

                done = done[0]

                if goal_reached:
                    done = True

                modified_states = []
                for state in current_state:
                    predicate, args = state.split('(', 1)
                    args = args.rstrip(').')
                    modified_state = f"{predicate}({args},s{state_id})."
                    modified_states.append(modified_state)
                
                # if reward >= 10:
                #     print(f"{state_id}   Modified states: {modified_states}")
                #     print(f"action: {action_key}")
                #     print(f"reward: {reward}")
                #     print(f"next_state: {next_state}")
                if state_id%1000 ==0:
                    save_image(img, f"state_{state_id}.png")

                trajectory.append((current_state, action, next_state, reward ,done)) 
                if done:                               
                    # trajectory.append((current_state, action, next_state, reward ,"Done"))  #check trajectory's format. whether the state format is consistent in all cases
                    traj_lens.append(traj_len)
                    next_state_qvalue = self.goal_qvalue
                    print(f"Reached goal, setting next state Q-value to goal Q-value: {next_state_qvalue}")
                else:
                    _, next_state_qvalue = self.get_action(next_state, q_function, best=True)     #env
                    traj_len += 1
                    if traj_len >= self.max_traj_len :
                        # trajectory.append((next_state, "END"))
                        traj_lens.append(traj_len)
                        done = True
                        # print("Reached max trajectory length, ending trajectory")
                    # else:
                current_state = next_state  


                q = ((1.0 - self.learning_rate) * q_value) + (self.learning_rate * (reward + (self.discount_factor * next_state_qvalue)))
                
                bellman_error.append(abs(q_value - (reward + (self.discount_factor * next_state_qvalue)))) #ToDo: bellman error is getting zero as both the rewards and q values are zero!!!!

                train_batch.facts += modified_states   
                train_batch.pos.append(f"regressionExample({action_key},{q[0]:.3f}).")

                state_id += 1
            new_batch.append(trajectory)
            log_trajectory(trajectory, filename="trajectory_log.json")

        return new_batch, state_id, traj_lens, bellman_error

    def get_training_batch(self, batch_size, q_function):
        replay_traj = []
        train_batch = Database()

        # I commented this so that we never use trajectories from previous iterations. why?
        # because It needs restoring the state with the previous state and we may do this later if necessary  # FIXME
        # if self.buffer.size > 0:  # randomely selects one of the trajectories. at first the size is 0
        #     # Sample historic trajectories
        #     # batch_size =10 , relay_sampling_rate = 0.10
        #     replay_traj = self.buffer.get_trajectories(int(self.replay_sampling_rate * batch_size))  #get a random number for this replay traj 

        gt.stamp('sampled historic trajectories', unique=False)
        new_traj, env_steps, traj_lens, bellman_error = self.generate_batch(train_batch, batch_size - len(replay_traj),
                                                                           q_function)
        state_id = env_steps
        gt.stamp('sampled new trajectories', unique=False)
        
        self.buffer.add_all_trajectories(new_traj)

        # We do not get pld traj from buffer, So we do not run this "for"
        for traj in replay_traj:
            # For all states in trajectory except last
            for i in range(len(traj) - 1):
                current_state, action, next_state = traj[i][0], traj[i][1], traj[i + 1][0]

                action_key = get_key_from_value(self.env.pred2action, action)
                action_key = "{MOVE}({state_id},{i})".format(MOVE = 'move', state_id=state_id, i= action_key)  #change stete_id and action here if getting here

                (next_state, _ ), reward, done, _ , _ = self.env.step(action)
                self.agent.logic_actor.compute_init_v(next_state)
                next_state , _= self.agent.logic_actor.print_valuations_input(self.agent.logic_actor.V_0, min_value=0.7)

                reward = torch.tensor(reward).to(self.device).view(-1)
                reward = reward.cpu().numpy()

                q_value = self.get_qvalue(current_state, action, q_function)      #env

                modified_states = []
                for state in current_state:
                    predicate, args = state.split('(', 1)
                    args = args.rstrip(').')
                    modified_state = f"{predicate}({args},s{state_id})." 
                    modified_states.append(modified_state)
                
                if traj[i+1][1] == "SUCCESS":
                    next_q_value = self.goal_qvalue
                else:
                     _, next_q_value = self.get_action(next_state, q_function, best=True)
                q = ((1.0 - self.learning_rate) * q_value) + self.learning_rate * (
                        reward + (self.discount_factor * next_q_value))
                bellman_error.append(abs(q_value - (reward + (self.discount_factor * next_q_value))))

                train_batch.facts += modified_states       
                train_batch.pos.append(f"regressionExample({action_key},{q_value:.3f}).")

                # self.add_sample(train_batch, current_state, action, q, state_id=str(state_id))
                state_id += 1
        
        gt.stamp('evaluate historic trajectories', unique=False)
        stats = create_stats_ordered_dict(
            'bellman error',
            bellman_error,
        )
        stats['batch size'] = batch_size
        stats['learning rate'] = self.learning_rate
        stats['discount factor'] = self.discount_factor
        stats['steps in env'] = env_steps
        stats['sample size'] = state_id
        stats['no of replay traj'] = len(replay_traj)
        stats['no of sampled traj'] = len(new_traj)
        return train_batch, stats

    def get_qvalue(self, state, action, q_function=None, env=None):
        if env is None:
            env = self.env         #env
        if q_function is None:
            return 0.0
        
        possible_actions = list(self.env.pred2action.values())           #env
    
        if action not in possible_actions:
            raise Exception("Invalid action")
        _, q_values, _ = self.predict(env, q_function, state, self.additional_facts)
        idx = possible_actions.index(action)
        return q_values[idx]
    

    def get_action(self, state, q_function=None, env=None, best=False):
        if env is None:
            env = self.env        #env
        # possible_actions = self.env.all_actions()   #based on getout  g
        possible_actions = list(env.pred2action.values()) #{'noop': 0, 'fire': 1, 'up': 2, 'right': 3, 'left': 4, 'down': 5}
        possible_actions = possible_actions[-4:] # not use fire or noop for now
        if q_function is None:    # at first q_function is none
            action = random.choice(possible_actions)    #this is where we should use Imitation learning for the initial q_function
            return action, 0.0
        
        idx, q_values, best_action = self.predict(env, q_function, state, self.additional_facts)   #env
        if not best:
            idx = self.exploration_strategy.get_action_idx(idx, len(possible_actions))
            # print(f"Exploration strategy selected action index: {idx}")
        return possible_actions[idx], q_values[idx]

    def train(self):
        """Fitted Q Learning"""
        current_q = None
        
        # current_q = RDNRegressor()
        # current_q.from_json("out/gbql-stack/gbql-stack_2025_04_16_19_20_43_0000--s-0/itr_20.json")
        

        if self.burn_in_traj > 0:   #This is zero in our case
            logger.log(f"adding {self.burn_in_traj} burn_in_traj")
            train_batch = Database()
            traj, _, _, _ = self.generate_batch(train_batch, self.burn_in_traj)
            self.buffer.add_all_trajectories(traj)
        
        logger.log("started fitted Q training")
        for i in gt.timed_for(range(self.n_iterations), save_itrs=True):
            logger.log(f"Iteration {i} started")
            logger.log(f"Iteration {i} getting training batch")
            train_batch, training_stats = self.get_training_batch(self.batch_size, current_q)
            gt.stamp("training batch", unique=False)
            logger.log(f"Iteration {i} fitting q function")

            updated_q = self.fit_q(train_batch, target="up,right,left,down",path=f"{logger.get_snapshot_dir()}/fitted-q/itr{i}", save=True) #returns a regressor
           
            
            gt.stamp("bsrl learning", unique=False)
        
            self.n_estimators.append(updated_q) 
            self.exploration_strategy.end_epoch()

            if i % self.test_gap == 0: # after 10 iterations it is evaluation time
                logger.log(f"Iteration {i} evaluating")
                paths = self.evaluate(self.n_eval_traj, updated_q)
                gt.stamp("bsrl evaluation", unique=False)
                self._log_stat(updated_q, training_stats, paths, i)
                logger.record_dict(self.exploration_strategy.stats(), prefix='exploration/')
                logger.dump_tabular()
                if self.learning_rate_strategy is not None:
                    self.learning_rate = self.learning_rate_strategy.end_epoch()
            current_q = updated_q
            
            # save_boosted_rdn_regressor(updated_q, filename=f"{logger.get_snapshot_dir()}/boosted_rdn_regressor.pkl")
            # save_external_files(updated_q, filename=f"{logger.get_snapshot_dir()}/boosted_rdn_regressor.pkl")
            
            logger.log(f"Iteration {i} ended")
        return current_q

    def _log_stat(self, q_function, training_stats, paths, itr):
        # for q_func in q_function:
        logger.save_itr_params(itr, q_function)
        logger.record_dict(training_stats, prefix='training/')
        buffer_stats = self.buffer.get_diagnostics()
        logger.record_dict(buffer_stats, prefix='buffer/')
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



    def evaluate(self, batch_size, q_function):
        """Evaluation in Test env """
        paths = []
        total_reward = 0
        # TODO: Evaluation takes maximum time, this could be improved with parallel environments
        for i in range(batch_size):
            total_reward = 0
            path = dict(states=[], actions=[], rewards=[], info=[],
                        episode_reward=0, is_success=False, episode_length=0, percent_solved=0.0)
            done = False
            test_state_logic , _ = self.test_env.reset() #FIXME: why do we need test env at all?
            (test_state_logic, _ ), _, _, _ , _ = self.env.step(0)
            # current_state = self.test_env.state
            self.agent.logic_actor.compute_init_v(test_state_logic)
            current_test_state, goal_reached = self.agent.logic_actor.print_valuations_input(self.agent.logic_actor.V_0, min_value=0.7)
            if goal_reached:
                done = True
                path['is_success'] = True

            traj_len = 0
            while not done:
                action, _ = self.get_action(current_test_state, q_function, env=self.test_env, best=True) 
                path['states'].append(current_test_state)
                action_key = get_key_from_value(self.env.pred2action, action)
                action_key = "{ACTION}".format(ACTION=action_key)
                path['actions'].append(action_key)
                
                (next_state, _ ), reward, done, _ , _ = self.test_env.step(action)
                self.agent.logic_actor.compute_init_v(next_state)
                next_symbolic, goal_reached = self.agent.logic_actor.print_valuations_input(self.agent.logic_actor.V_0, min_value=0.7)
                if goal_reached:
                    done = True
               
                r = torch.tensor(reward).to(self.device).view(-1)
                r = r.cpu().numpy()

                if current_test_state == next_symbolic:
                    r -= 1

                total_reward += r
                path['rewards'].append(r)
                path['episode_reward'] += total_reward
                # if current_state == next_state:
                #     traj_len = self.max_traj_len
                traj_len += 1
                # solved = self.test_env.is_goal_state(next_state)
                solved = done
                if solved or traj_len >= self.max_traj_len:
                    path['states'].append(next_symbolic)
                    path['is_success'] = solved
                    path['episode_length'] = traj_len
                    # path['percent_solved'] = self.test_env.get_solved_percent()
                    path['percent_solved'] = ''  #FIXME
                    done = True
                else:
                    current_test_state = next_symbolic
            paths.append(path)
            print(path)
        return paths


    def predict(self, env, q_function, state, additional_facts=None):   
        # check cache
        # q_values = q_function.fetch_cache(state)
        q_values = None   #FIXME: comes from env.state_space problem
        state_id = "s1"
        all_actions = list(env.pred2action.values())[-4:]
        if q_values is not None:
            # all_actions = env.all_actions(state, state_id="s1", regression=True)
            where_max = np.where(q_values == np.max(q_values))[0]
            if len(where_max) == 1:
                idx = where_max[0]
            else:
                idx = np.random.choice(where_max)
            return idx, q_values, all_actions[idx]
        
        test = Database()
        modified_test_states = []
        for state in state:
            predicate, args = state.split('(', 1)
            args = args.rstrip(').')
            modified_state = f"{predicate}({args},s{state_id})."
            modified_test_states.append(modified_state)

        test.facts = modified_test_states
        # print("modified_test_states: ", modified_test_states)
        
        q_test_values = 0.0        # it use this as true value. In block it is always zero
        for action in all_actions:
            action_key = get_key_from_value(self.env.pred2action, action)
            action_key = "{ACTION}({player},{state_id})".format(ACTION = action_key, player = "obj1", state_id=f"s{state_id}")
            test.pos.append(f"regressionExample({action_key},{q_test_values:.3f}).")
        
        if additional_facts is not None:
            test.facts += additional_facts

        # q_values = []
        # for q_func in q_function:

        q_values = q_function.predict(test)     #rql prediction
        # print("q_values: ", q_values)  #up,down,left,right

        where_max = np.where(q_values == np.max(q_values))[0]
        if len(where_max) == 1:
            idx = where_max[0]
        else:
            idx = np.random.choice(where_max)
        return idx, q_values, test.pos[idx]


class RRT(GBQL):
    """The RRT code is same as GBQL but with only 1 tree"""

    #I changed goal_q_value to 1. but this is different in kangaroo. As we dont know when we reached the goal!!! Done is not equivalent
    #to goal or success in the env.
    def __init__(self, n_iter=1, batch_size=10, train_env=None, bk=None, agent=None, max_trajectory_length=50,
                 replay_sampling_rate=0.10, test_env=None, max_buffer_size=1000, target_predicate="q_value",
                 learning_rate=0.9, discount_factor=0.99, n_evaluation_trajectories=10,
                 n_burn_in_traj=0, additional_facts=None, goal_q_value=100, exploration_strategy=EpsilonGreedy(), device=None,
                 learning_rate_strategy=None, buffer=ReplayBuffer, test_gap=10):
        super().__init__(n_iter=n_iter, n_trees=1, batch_size=batch_size, train_env=train_env, bk=bk,
                         max_trajectory_length=max_trajectory_length, replay_sampling_rate=replay_sampling_rate,
                         test_env=test_env, agent=agent, max_buffer_size=max_buffer_size, target_predicate=target_predicate,
                         learning_rate=learning_rate, discount_factor=discount_factor,
                         n_evaluation_trajectories=n_evaluation_trajectories, n_burn_in_traj=n_burn_in_traj,
                         additional_facts=additional_facts, goal_q_value=goal_q_value,
                         exploration_strategy=exploration_strategy, device = device, learning_rate_strategy=learning_rate_strategy,
                         buffer=buffer, test_gap=test_gap)


