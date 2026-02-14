import os
import json
import numpy as np
import cv2
import time
from datetime import datetime
import argparse
import sys

class RandomAgent:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = None
        self.action_space = None
        self.rgb_key = 'head_rgb' # Default

    def setup_env(self):
        print(f"Setting up environment: {self.env_name}")
        if self.env_name == "EB-ALFRED":
            from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv
            self.env = EBAlfEnv(eval_set='base', selected_indexes=[0], resolution=300)
            self.rgb_key = 'head_rgb'
        elif self.env_name == "EB-Manipulation":
            from embodiedbench.envs.eb_manipulation.EBManEnv import EBManEnv
            self.env = EBManEnv(eval_set='base', selected_indexes=[0], img_size=(300, 300))
            self.rgb_key = 'front_rgb'
        elif self.env_name == "EB-Navigation":
            from embodiedbench.envs.eb_navigation.EBNavEnv import EBNavigationEnv
            self.env = EBNavigationEnv(eval_set="base", selected_indexes=[0], resolution=300)
            self.rgb_key = 'head_rgb'
        elif self.env_name == "EB-Habitat":
            from embodiedbench.envs.eb_habitat.EBHabEnv import EBHabEnv
            self.env = EBHabEnv(eval_set="base")
            self.rgb_key = 'head_rgb'
        else:
            raise ValueError(f"Unknown environment: {self.env_name}")
        
        self.action_space = self.env.action_space

    def get_action(self):
        return self.action_space.sample()

    def get_frame(self, obs):
        if isinstance(obs, dict):
            if self.rgb_key in obs:
                 return np.array(obs[self.rgb_key]).astype(np.uint8)
            # Fallback check
            for key in ['head_rgb', 'front_rgb', 'rgb']:
                if key in obs:
                    return np.array(obs[key]).astype(np.uint8)
        return None

    def save_video(self, frames, output_path, fps=10):
        if not frames:
            return
        height, width, layers = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in frames:
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video.release()

    def run_episode(self, num_steps=30):
        if not self.env:
            self.setup_env()

        # Create logs directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"random_agent_logs/{self.env_name}/{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        
        # Reset
        if self.env_name == "EB-Manipulation":
            desc, obs = self.env.reset()
            instruction = desc
        else:
            obs = self.env.reset()
            instruction = getattr(self.env, "episode_language_instruction", "No instruction")

        logs = {
            "env_name": self.env_name,
            "instruction": instruction,
            "steps": []
        }
        
        frames = []
        frame = self.get_frame(obs)
        if frame is not None:
            frames.append(frame)
            # Save initial frame
            init_frame_path = os.path.join(log_dir, "step_0.png")
            cv2.imwrite(init_frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        else:
            init_frame_path = None

        # Log initial state
        logs["steps"].append({
            "step": 0,
            "action_id": None,
            "action_description": "Initial State",
            "reward": 0.0,
            "done": False,
            "info": {},
            "frame_path": init_frame_path
        })

        for i in range(1, num_steps + 1):
            action = self.get_action()
            
            # Environment nuances
            if self.env_name == "EB-Manipulation":
                if isinstance(action, (np.ndarray, list)):
                     action[-1] = 1.0 # Ensure gripper state valid
                obs, reward, done, info = self.env.step(action)
                action_desc = "Continuous Action"
            elif self.env_name == "EB-Navigation":
                obs, reward, done, info = self.env.step(action, "", 0)
                action_desc = self.env.language_skill_set[action]
            elif self.env_name == "EB-ALFRED":
                obs, reward, done, info = self.env.step(action)
                # EBAlfEnv: action is int index
                action_desc = self.env.language_skill_set[action] if hasattr(self.env, 'language_skill_set') else str(action)
            elif self.env_name == "EB-Habitat":
                obs, reward, done, info = self.env.step(action)
                 # EBHabEnv: action is int index
                action_desc = self.env.language_skill_set[action] if hasattr(self.env, 'language_skill_set') else str(action)
            else:
                obs, reward, done, info = self.env.step(action)
                action_desc = str(action)

            # Capture Frame
            frame = self.get_frame(obs)
            frame_path = None
            if frame is not None:
                frames.append(frame)
                frame_path = os.path.join(log_dir, f"step_{i}.png")
                cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Log Step
            step_data = {
                "step": i,
                "action_id": int(action) if isinstance(action, (int, np.integer)) else action.tolist(),
                "action_description": action_desc,
                "reward": float(reward),
                "done": bool(done),
                "info": str(info),
                "frame_path": frame_path
            }
            logs["steps"].append(step_data)
            
            if done:
                print(f"Episode finished at step {i}")
                break

        # Save artifacts
        log_path = os.path.join(log_dir, "log.json")
        video_path = os.path.join(log_dir, "video.mp4")
        
        with open(log_path, 'w') as f:
            json.dump(logs, f, indent=4)
        
        self.save_video(frames, video_path)
        print(f"Saved log to {log_path}")
        print(f"Saved frames to {log_dir}")
        print(f"Saved video to {video_path}")
        
        self.env.close()

def main():
    parser = argparse.ArgumentParser(description="Run random agent on EmbodiedBench environments")
    parser.add_argument("--env", type=str, required=True, 
                        choices=["EB-ALFRED", "EB-Manipulation", "EB-Navigation", "EB-Habitat"],
                        help="Environment to run")
    args = parser.parse_args()

    agent = RandomAgent(args.env)
    try:
        agent.run_episode()
    except Exception as e:
        print(f"Error running agent on {args.env}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
