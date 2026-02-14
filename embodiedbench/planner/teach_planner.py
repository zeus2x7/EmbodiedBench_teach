
import re
import os
import numpy as np
import cv2
import json
from embodiedbench.planner.planner_utils import local_image_to_data_url, truncate_message_prompts
from embodiedbench.planner.remote_model import RemoteModel
from embodiedbench.planner.custom_model import CustomModel
from embodiedbench.planner.planner_utils import template, template_lang
from embodiedbench.main import logger

MESSAGE_WINDOW_LEN = 10

class EBTeachPlanner():
    def __init__(self, model_name = '', model_type = 'remote', actions = [], system_prompt = '', examples = '', n_shot=1, obs_key='head_rgb', chat_history=False, language_only=False, multiview = False, multistep = False, visual_icl = False, tp=1, truncate=False, kwargs={}):
        self.model_name = model_name
        self.model_type = model_type
        self.obs_key = obs_key
        self.system_prompt = system_prompt
        self.n_shot = n_shot
        self.chat_history = chat_history # whether to include all the chat history for prompting
        self.truncate = truncate
        self.set_actions(actions)
        self.planner_steps = 0
        self.output_json_error = 0
        self.kwargs = kwargs
        self.action_key = kwargs.pop('action_key', 'action_id')
        self.multiview = multiview
        self.multistep = multistep
        self.visual_icl = visual_icl
        
        self.examples = examples[:n_shot]
        self.language_only = language_only

        if model_type == 'custom':
            self.model = CustomModel(model_name, language_only)
        else:
            self.model = RemoteModel(model_name, model_type, language_only, tp=tp)

    def set_actions(self, actions):
        self.actions = actions
        self.available_action_str = self.get_available_action_prompt(actions)

    def get_available_action_prompt(self, available_actions):
        available_action_str = ''
        for i in range(len(available_actions)):
            available_action_str += '\naction id ' + str(i) + ': ' + str(available_actions[i]) 
            if i < len(available_actions) - 1:
                available_action_str += ', '
        return available_action_str

    def process_prompt(self, user_instruction, prev_act_feedback=[]):
        user_instruction = user_instruction.rstrip('.')
        
        # Simple prompt construction as seen in other planners
        if len(prev_act_feedback) == 0:
            if self.n_shot >= 1 and self.examples:
                 prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example {i}: \n {x}' for i,x in enumerate(self.examples)])) 
            else:
                 prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')
            
            prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
            prompt += f"\nYou are supposed to output in JSON.{template_lang if self.language_only else template}"

        else:
             # Basic history handling
            if self.n_shot >= 1 and self.examples:
                 prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example {i}: \n {x}' for i,x in enumerate(self.examples)]))
            else:
                 prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')

            prompt += f'\n\n## The human instruction is: {user_instruction}.'
            prompt += '\n\n The action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                 prompt += '\n Step {}, action id {}, {}, env feedback: {}'.format(i, action_feedback[0], self.actions[action_feedback[0]], action_feedback[1])
            
            prompt += f"\nYou are supposed to output in JSON.{template_lang if self.language_only else template}"
        
        return prompt

    def get_message(self, image, prompt, messages=[]):
        if self.language_only:
             current_message = {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        else:
             # Standard image handling
            data_url = local_image_to_data_url(image_path=image)
            current_message = {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    }, 
                    {"type": "text", "text": prompt}
                ],
            }
        
        messages = messages + [current_message]
        return messages[-MESSAGE_WINDOW_LEN:]

    def reset(self):
        self.episode_messages = []
        self.episode_act_feedback = []
        self.planner_steps = 0
        self.output_json_error = 0

    def json_to_action(self, output_text, json_key='executable_plan'):
        valid = True
        try:
            # Basic cleanup
            output_text = output_text.replace("'",'"').replace('```json', '').replace('```', '')
            json_object = json.loads(output_text)
            action = [x[self.action_key] for x in json_object[json_key]]
            if not len(action):
                print('empty plan, using random action instead')
                action = np.random.randint(len(self.actions))
        except Exception as e:
            print("Failed to decode JSON:", e)
            print('random action')
            self.output_json_error += 1
            action = np.random.randint(len(self.actions))
            valid = False
        return action, valid

    def act(self, observation, user_instruction):
        if type(observation) == dict:
            obs = observation[self.obs_key]
        else:
            obs = observation # input image path
        
        prompt = self.process_prompt(user_instruction, prev_act_feedback=self.episode_act_feedback)
        
        if len(self.episode_messages) == 0:
             self.episode_messages = self.get_message(obs, prompt)
        else:
            if self.chat_history:
                self.episode_messages = self.get_message(obs, prompt, self.episode_messages)
            else:
                self.episode_messages = self.get_message(obs, prompt)
        
        messages_to_send = self.episode_messages
        if self.chat_history and self.truncate:
            messages_to_send = truncate_message_prompts(self.episode_messages)

        try:
            out = self.model.respond(messages_to_send)
        except Exception as e:
            print(f"Model error: {e}")
            out = "{}" # Will fail json decode and trigger random action
            
        if self.chat_history:
            self.episode_messages.append(
                {
                "role": "assistant",
                "content": [{"type": "text", "text": out}],
                }
            )
            
        logger.debug(f"Model Output:\n{out}\n")
        action, valid = self.json_to_action(out)
        self.planner_steps += 1
        
        # Return action (single or list) and raw output
        # If action is a list of length 1, return the item
        if isinstance(action, list) and len(action) == 1:
             action = action[0]

        return action, out

    def update_info(self, info):
        self.episode_act_feedback.append([
            info.get('action_id', -1), # Fallback if action_id missing
            info.get('env_feedback', '') # Fallback
        ])
