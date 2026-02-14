import os
import numpy as np
from tqdm import tqdm
import time
import json
from embodiedbench.envs.eb_teach.EBTeachEnv import EBTeachEnv
from embodiedbench.planner.teach_planner import EBTeachPlanner
from embodiedbench.evaluator.summarize_result import average_json_values
from embodiedbench.evaluator.evaluator_utils import update_config_with_args
from embodiedbench.evaluator.config.system_prompts import eb_teach_system_prompt
from embodiedbench.main import logger

class EB_TeachEvaluator():
    def __init__(self, config):
        self.model_name = config['model_name']
        self.eval_set = config.get('eval_set', 'valid_seen')
        self.config = config
        self.env = None
        self.planner = None

    def save_episode_metric(self, episode_info):
        # EBTeachEnv doesn't have current_episode_num attribute exposed directly in same way as Alfred
        # We might need to handle this. For now assuming sequential execution.
        episode_idx = self.env.current_instance['instance_id'] if self.env.current_instance else "unknown"
        filename = 'episode_{}_final_res.json'.format(episode_idx)
        res_path = os.path.join(self.result_path, 'results')
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        with open(os.path.join(res_path, filename), 'w', encoding='utf-8') as f:
            json.dump(episode_info, f, ensure_ascii=False)

    def evaluate_main(self):
        # Setup logging path
        exp_name = f"{self.model_name.split('/')[-1]}_{self.config['exp_name']}/{self.eval_set}" if len(self.config['exp_name']) else f"{self.model_name.split('/')[-1]}/{self.eval_set}"
        self.result_path = os.path.join('evaluate_results/eb_teach', exp_name)
        if not os.path.exists(self.result_path):
             os.makedirs(self.result_path)

        logger.info(f'Current eval set: {self.eval_set}')
        
        self.env = EBTeachEnv(split=self.eval_set, resolution=self.config.get('resolution', 300))
        
        # We don't have explicit examples json for TEACh yet, passing empty list or need to create one
        examples = [] 
        
        self.planner = EBTeachPlanner(
            model_name=self.model_name, 
            model_type=self.config.get('model_type', 'remote'), 
            actions=self.env.language_skill_set, 
            system_prompt=eb_teach_system_prompt, 
            examples=examples, 
            n_shot=self.config['n_shots'], 
            obs_key='head_rgb', 
            chat_history=self.config['chat_history'], 
            language_only=self.config['language_only'],
            multistep=self.config.get('multistep', 0), 
            tp=self.config.get('tp', 1)
        )

        self.evaluate()
        average_json_values(os.path.join(self.result_path, 'results'), output_file='summary.json')
        with open(os.path.join(self.result_path, 'config.txt'), 'w') as f:
            f.write(str(self.config))

    def evaluate(self):
        # We process a fixed number of episodes or until env exhaustion
        # specific to how EBTeachEnv is implemented (random sampling for now in my implemention)
        # To make it deterministic we might want to iterate through files. 
        # But EBTeachEnv as implemented in previous step does random sampling.
        # Let's assume we run for 50 episodes for now as per user request flow usually
        num_episodes = 20 
        progress_bar = tqdm(total=num_episodes, desc="Episodes")
        
        for i in range(num_episodes):
            logger.info(f"Evaluating episode {i} ...")
            episode_info = {'reward': [], 'num_invalid_actions': 0}
            obs = self.env.reset()
            # We need a save_image method or manually save
            # For simplicity, we skip saving every frame to disk unless planner needs path
            # The planner expects path if remote model, or we can pass np array if local?
            # VLMPlanner logic expects path usually.
            
            img_path = os.path.join(self.result_path, f"ep_{i}_start.png")
            import cv2
            if 'head_rgb' in obs:
                 cv2.imwrite(img_path, cv2.cvtColor(obs['head_rgb'], cv2.COLOR_RGB2BGR))
            
            # Use instruction from environment
            user_instruction = getattr(self.env, 'episode_language_instruction', "Interact with the environment to complete the task.")
            logger.info(f"Instruction: {user_instruction}")

            self.planner.reset()
            done = False
            step_count = 0
            max_steps = 50
            
            while not done and step_count < max_steps:
                try:
                    action_idx, reasoning = self.planner.act(img_path, user_instruction)
                    print(f"Planner Output Action: {action_idx}")
                    
                    if action_idx == -1: # invalid
                         # Handle invalid
                         pass
                    
                    # Execute
                    obs, reward, done, info = self.env.step(action_idx)
                    
                    # Save new image
                    img_path = os.path.join(self.result_path, f"ep_{i}_step_{step_count}.png")
                    if 'head_rgb' in obs:
                        cv2.imwrite(img_path, cv2.cvtColor(obs['head_rgb'], cv2.COLOR_RGB2BGR))

                    episode_info['reward'].append(reward)
                    step_count += 1
                    
                except Exception as e:
                    print(e)
                    break
            
            # Log results
            episode_info['task_success'] = info.get('task_success', False)
            episode_info['goal_condition_success'] = info.get('goal_condition_success', 0.0)
            self.save_episode_metric(episode_info)
            progress_bar.update()

if __name__ == '__main__':
    import argparse
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Change configuration parameters.')
        parser.add_argument('--model_name', type=str, default='gpt-4o-mini', help='Name of the model.')
        parser.add_argument('--n_shots', type=int, default=10, help='Number of examples')
        parser.add_argument('--model_type', type=str, default='remote', help='Type of the model.')
        parser.add_argument('--language_only', type=int, default=0, help='Set to True for language only mode.')
        parser.add_argument('--exp_name', type=str, default='teach_eval', help='Name of the experiment.')
        parser.add_argument('--chat_history', type=int, default=0, help='Set to True to enable chat history.')
        parser.add_argument('--eval_set', type=str, default='valid_seen', help='Evaluation set.')
        parser.add_argument('--multistep', type=int, default=0, help='Number of steps for multi-step reasoning.')
        parser.add_argument('--resolution', type=int, default=300, help='Resolution for processing.')
        parser.add_argument('--tp', type=int, default=1, help='number of tensor parallel splits of the model parameters')
        return parser.parse_args()

    args = parse_arguments()
    config = vars(args)

    evaluator = EB_TeachEvaluator(config)
    evaluator.evaluate_main()
