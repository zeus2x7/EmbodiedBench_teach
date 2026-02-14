import gym
import numpy as np
import os
import random
import json
import glob
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import time

# TEACh imports (will work in embench_teach env)
try:
    from teach.inference.inference_runner_base import InferenceRunnerConfig
    from teach.inference.edh_inference_runner import EdhInferenceRunner
    from teach.replay.episode_replay import EpisodeReplay
    from teach.logger import create_logger
    from teach.utils import load_json
    from teach.inference.actions import obj_interaction_actions, all_agent_actions
except ImportError as e:
    print(f"Warning: teach not found. Make sure you are in the 'embench_teach' environment. Error: {e}")
    # Define fallbacks to avoid NameError during init if teach is missing
    obj_interaction_actions = []
    all_agent_actions = []

class EBTeachEnv(gym.Env):
    def __init__(self, data_dir=None, split='valid_seen', resolution=300):
        self.resolution = resolution
        self.split = split
        # Default data dir relative to this file or specific path
        if data_dir is None:
            # Assuming teach repo is cloned at EmbodiedBench/teach
            # and data is in EmbodiedBench/teach/teach-dataset (after teach_download)
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            possible_paths = [
                os.path.join(base_path, "teach", "teach-dataset"),
                "/tmp/teach-dataset"
            ]
            self.data_dir = None
            for p in possible_paths:
                if os.path.exists(p):
                    self.data_dir = p
                    break
            
            if self.data_dir is None:
                 print(f"Warning: TEACh data directory not found in {possible_paths}. defaulting to {possible_paths[0]}")
                 self.data_dir = possible_paths[0]
        else:
            self.data_dir = data_dir

        if not os.path.exists(self.data_dir):
            print(f"Warning: TEACh data directory not found at {self.data_dir}")
        
        # Load EDH instances
        self.edh_instance_files = self._get_edh_files()
        
        # Action space: Discrete index mapping to all_agent_actions
        self.actions = all_agent_actions
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.language_skill_set = self.actions

        self.er = None
        self.current_instance = None
        self.config = InferenceRunnerConfig(
            data_dir=self.data_dir,
            split=self.split,
            output_dir="/tmp/teach_output",
            images_dir="/tmp/teach_images",
            model_class=None, # Not used here
            model_args=[],
        )

    def _get_edh_files(self):
        edh_dir = os.path.join(self.data_dir, "edh_instances", self.split)
        if not os.path.exists(edh_dir):
             return []
        return glob.glob(os.path.join(edh_dir, "*.json"))

    def _init_episode_replay(self):
        if self.er is None:
             # Initialize EpisodeReplay with x_display from env var or default
             # AI2-THOR requires X server (handled by startx script)
             self.er = EpisodeReplay("thor", ["ego", "allo", "targetobject"])

    def reset(self):
        self._init_episode_replay()
        
        if not self.edh_instance_files:
            print("No EDH instances found. Returning empty obs.")
            return {'head_rgb': np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)}

        # underlying logic from EdhInferenceRunner._run_instance
        max_retries = 3
        # Iteration logic
        if not hasattr(self, 'current_episode_idx'):
            self.current_episode_idx = 0
            
        for _ in range(max_retries):
            # Sequential iteration
            if self.current_episode_idx >= len(self.edh_instance_files):
                self.current_episode_idx = 0
                
            instance_file = self.edh_instance_files[self.current_episode_idx]
            self.current_episode_idx += 1
            
            instance = load_json(instance_file)
            self.current_instance = instance
            
            game_file = os.path.join(
                self.data_dir, "games", self.split, f"{instance['game_id']}.game.json"
            )
            
            # Helper from EdhInferenceRunner to calculate state diff task
            check_task = EdhInferenceRunner._get_check_task(instance, self.config)

            # Replay history
            success, self.er = EdhInferenceRunner._initialize_episode_replay(
                instance, game_file, check_task, 
                replay_timeout=500, er=self.er
            )
            
            if success:
                # Load history images (optional, but good for context if we were a real agent)
                # EdhInferenceRunner._maybe_load_history_images(instance, self.config)
                
                # Get initial observation
                images = self.er.simulator.get_latest_images()
                obs = {}
                if "ego" in images:
                    obs['head_rgb'] = images["ego"]
                else:
                    obs['head_rgb'] = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
                
                # Get instruction
                # EDH instance has "dialog_history": [{"speaker": "Driver", "text": "..."}]
                # We want the last instruction from the "Commander" or similar
                # For simplicity, we join all text or take the last one.
                # In TEACh, it's usually dialog history.
                instruction = ""
                if 'dialog_history' in instance:
                    instruction = " ".join([x['text'] for x in instance['dialog_history']])
                
                self.episode_language_instruction = instruction
                return obs
        
        print("Failed to reset after retries.")
        return {'head_rgb': np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)}

    def step(self, action_idx):
        if self.er is None:
            return self.reset(), 0, True, {}

        action_name = self.actions[action_idx] if isinstance(action_idx, int) else action_idx
        
        # Handle object interaction coordinates
        # For random agent, we pick random coordinates
        # For real agents, this info should be passed, but Gym step takes 1 arg usually.
        # We assume 0.5, 0.5 (center) for now or random
        obj_relative_coord = [0.5, 0.5]
        if action_name in obj_interaction_actions:
             obj_relative_coord = [random.random(), random.random()]

        # Execute action using logic from InferenceRunnerBase
        # _execute_action(simulator, action, obj_relative_coord)
        
        if action_name == "Stop":
            step_success = True
            done = True
        else:
            if action_name in obj_interaction_actions:
                step_success, _, _ = self.er.simulator.apply_object_interaction(action_name, 1, obj_relative_coord[1], obj_relative_coord[0])
            else:
                step_success, _, _ = self.er.simulator.apply_motion(action_name, 1)
            done = False
        
        # Get Obs
        images = self.er.simulator.get_latest_images()
        obs = {}
        if "ego" in images:
            obs['head_rgb'] = images["ego"]
        else:
            obs['head_rgb'] = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)

        # Metrics calculation
        task_desc, success, subgoals, gc_total, gc_satisfied = self.er.simulator.check_episode_progress(self.er.simulator.current_task)
        
        reward = 0.0 # Dense reward could be implemented based on gc_satisfied change
        
        info = {
            'success': step_success, # Action success
            'task_success': success, # Episode success
            'goal_condition_success': gc_satisfied / gc_total if gc_total > 0 else 1.0,
            'action': action_name,
            'coord': obj_relative_coord,
            'instruction': self.episode_language_instruction
        }
        
        # Auto-finish if success
        if success:
             done = True

        return obs, reward, done, info

    def close(self):
        if self.er:
            self.er.simulator.shutdown_simulator()
