import os
import sys
import json
import numpy as np
import cv2
import time
from datetime import datetime
import argparse
import random
import glob

# Add teach src to path for native TEACh imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "teach_integrate", "teach", "src"))


class RandomAgent:
    def __init__(self, env_name, teach_data_dir=None):
        self.env_name = env_name
        self.env = None
        self.action_space = None
        self.rgb_key = 'head_rgb'  # Default for EmbodiedBench envs

        # TEACh-specific state (native API)
        self.teach_data_dir = teach_data_dir
        self.er = None           # EpisodeReplay instance
        self.game_file = None
        self.game_data = None

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
        elif self.env_name == "EB-TEACh":
            self._setup_teach()
        else:
            raise ValueError(f"Unknown environment: {self.env_name}")

        if self.env_name != "EB-TEACh":
            self.action_space = self.env.action_space

    # ------------------------------------------------------------------ #
    #  TEACh-specific setup (native teach API, no EBTeachEnv wrapper)     #
    # ------------------------------------------------------------------ #
    def _setup_teach(self):
        from teach.replay.episode_replay import EpisodeReplay
        from teach.inference.actions import all_agent_actions, obj_interaction_actions

        # Store action lists for later use
        self._teach_actions = all_agent_actions
        self._teach_obj_actions = obj_interaction_actions

        data_dir = self.teach_data_dir or os.path.join(
            os.path.dirname(__file__), "teach_integrate", "teach", "data"
        )
        print(f"  TEACh data dir: {data_dir}")

        # Ensure DISPLAY is set for ai2thor
        if "DISPLAY" not in os.environ:
            os.environ["DISPLAY"] = ":1"
            print("  Set DISPLAY=:1")

        self.er = EpisodeReplay("thor", ["ego", "allo", "targetobject"])

        # Find a valid game file
        game_files = glob.glob(os.path.join(data_dir, "games", "valid_seen", "*.game.json"))
        if not game_files:
            game_files = glob.glob(os.path.join(data_dir, "**", "*.game.json"), recursive=True)
        if not game_files:
            raise FileNotFoundError(f"No game files found in {data_dir}")

        self.game_file = random.choice(game_files)
        print(f"  Selected game file: {self.game_file}")

        self.er.set_episode_by_fn_and_idx(self.game_file, 0, 0)

        with open(self.game_file) as f:
            self.game_data = json.load(f)

    def _start_teach_episode(self):
        """Launch the ai2thor simulator for the loaded TEACh game."""
        try:
            task = self.game_data['tasks'][0]
            world = (task.get('world') or task.get('scene') or task.get('floor_plan'))
            if not world and hasattr(self.er.episode, 'world'):
                world = self.er.episode.world
            if not world:
                defs = self.game_data.get('definitions', {})
                world = defs.get('world') or defs.get('scene_name')
            if not world:
                print("  Could not find world/scene name. Using default FloorPlan1.")
                world = "FloorPlan1"
            world_type = task.get('world_type', 'standard')
        except Exception as e:
            print(f"  Error accessing task/world: {e}")
            world = "FloorPlan1"
            world_type = "standard"

        print(f"  Starting world: {world}")
        self.er.simulator.start_new_episode(world=world, world_type=world_type)

    def _get_teach_frame(self):
        """Get the latest ego-centric image from the TEACh simulator."""
        images = self.er.simulator.get_latest_images()
        if 'ego' in images:
            return np.array(images['ego']).astype(np.uint8)
        return None

    def _step_teach(self):
        """Take a random action in the TEACh simulator. Returns (action_desc, success, info)."""
        action = random.choice(self._teach_actions)
        info = {}
        if action in self._teach_obj_actions:
            x, y = random.random(), random.random()
            success, _, _ = self.er.simulator.apply_object_interaction(action, 1, x, y)
            info['coord'] = [x, y]
        else:
            success, _, _ = self.er.simulator.apply_motion(action, 1)
        return action, success, info

    # ------------------------------------------------------------------ #
    #  Generic helpers                                                     #
    # ------------------------------------------------------------------ #
    def get_action(self):
        return self.action_space.sample()

    def get_frame(self, obs):
        if isinstance(obs, dict):
            if self.rgb_key in obs:
                return np.array(obs[self.rgb_key]).astype(np.uint8)
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

    # ------------------------------------------------------------------ #
    #  Run episode                                                         #
    # ------------------------------------------------------------------ #
    def run_episode(self, num_steps=30):
        if self.env_name == "EB-TEACh":
            return self._run_teach_episode(num_steps)
        return self._run_embench_episode(num_steps)

    # -- EmbodiedBench envs (ALFRED, Habitat, Navigation, Manipulation) -- #
    def _run_embench_episode(self, num_steps=30):
        if not self.env:
            self.setup_env()

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
            cv2.imwrite(os.path.join(log_dir, "step_0.png"),
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            init_frame_path = os.path.join(log_dir, "step_0.png")
        else:
            init_frame_path = None

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

            if self.env_name == "EB-Manipulation":
                if isinstance(action, (np.ndarray, list)):
                    action[-1] = 1.0
                obs, reward, done, info = self.env.step(action)
                action_desc = "Continuous Action"
            elif self.env_name == "EB-Navigation":
                obs, reward, done, info = self.env.step(action, "", 0)
                action_desc = self.env.language_skill_set[action]
            elif self.env_name == "EB-ALFRED":
                obs, reward, done, info = self.env.step(action)
                action_desc = (self.env.language_skill_set[action]
                               if hasattr(self.env, 'language_skill_set') else str(action))
            elif self.env_name == "EB-Habitat":
                obs, reward, done, info = self.env.step(action)
                action_desc = (self.env.language_skill_set[action]
                               if hasattr(self.env, 'language_skill_set') else str(action))
            else:
                obs, reward, done, info = self.env.step(action)
                action_desc = str(action)

            frame = self.get_frame(obs)
            frame_path = None
            if frame is not None:
                frames.append(frame)
                frame_path = os.path.join(log_dir, f"step_{i}.png")
                cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

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
            print(f"Step {i}: {action_desc}  reward={reward}  done={done}")

            if done:
                print(f"Episode finished at step {i}")
                break

        self._save_artifacts(log_dir, logs, frames)
        self.env.close()

    # -- TEACh (native teach API) ---------------------------------------- #
    def _run_teach_episode(self, num_steps=30):
        if not self.er:
            self.setup_env()

        self._start_teach_episode()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"random_agent_logs/EB-TEACh/{timestamp}"
        os.makedirs(log_dir, exist_ok=True)

        logs = {
            "env_name": "EB-TEACh",
            "game_file": self.game_file,
            "steps": []
        }

        frames = []

        # Initial frame
        frame = self._get_teach_frame()
        if frame is not None:
            frames.append(frame)
            cv2.imwrite(os.path.join(log_dir, "step_0.png"),
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        logs["steps"].append({
            "step": 0,
            "action_id": None,
            "action_description": "Initial State",
            "reward": 0.0,
            "done": False,
            "info": {},
            "frame_path": os.path.join(log_dir, "step_0.png")
        })

        for i in range(1, num_steps + 1):
            action, success, info = self._step_teach()

            frame = self._get_teach_frame()
            frame_path = None
            if frame is not None:
                frames.append(frame)
                frame_path = os.path.join(log_dir, f"step_{i}.png")
                cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            logs["steps"].append({
                "step": i,
                "action_id": action,
                "action_description": action,
                "reward": 0.0,
                "done": False,
                "info": str(info),
                "frame_path": frame_path,
                "success": success
            })
            print(f"Step {i}: Action={action}, Success={success}")

        self._save_artifacts(log_dir, logs, frames)
        self.er.simulator.shutdown_simulator()

    # -- Shared save ------------------------------------------------------ #
    def _save_artifacts(self, log_dir, logs, frames):
        log_path = os.path.join(log_dir, "log.json")
        video_path = os.path.join(log_dir, "video.mp4")
        with open(log_path, 'w') as f:
            json.dump(logs, f, indent=4)
        self.save_video(frames, video_path)
        print(f"Saved log   -> {log_path}")
        print(f"Saved video -> {video_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run random agent on EmbodiedBench environments (including TEACh)")
    parser.add_argument("--env", type=str, required=True,
                        choices=["EB-ALFRED", "EB-Manipulation", "EB-Navigation",
                                 "EB-Habitat", "EB-TEACh"],
                        help="Environment to run")
    parser.add_argument("--teach_data_dir", type=str, default=None,
                        help="TEACh data directory (default: teach_integrate/teach/data)")
    parser.add_argument("--steps", type=int, default=30,
                        help="Number of steps per episode")
    args = parser.parse_args()

    agent = RandomAgent(args.env, teach_data_dir=args.teach_data_dir)
    try:
        agent.run_episode(num_steps=args.steps)
    except Exception as e:
        print(f"Error running agent on {args.env}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
