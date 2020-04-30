import gym
import math
import os, time
from gym import error, spaces, utils, logger
from gym.utils import seeding
import numpy as np
from PIL import Image

class MinecraftLiveEnv(gym.Env):
  """
  Description:
      Vanilla Minecraft Java Edition - This gym was designed to work tightly with Minecraft as players would play it, unlike tools such as project Malmo 
      that rely on plugins to interact with the environment. With MinecraftLive, the agent is exposed to the Minecraft environment by sending keyboard and 
      mouse emulations to the environment and capturing/processing a screenshot as part of the observation process.

      Currently integrates with the MindcraftMind project, a tool designed to interact with the Minecraft client without the use of plugins. 
      TODO: Adapt the environment to be configurable to be used with other agents, such as Malmo.
  Source:
      Project Malmo and others involving the applications of RL to Minecraft
  Observation: 
      Type: Agent analyzes the image and captures health, food, brightness data.        
  Actions:
      Type: Commands to send to the Minecraft client via the agent (currently MindcraftMind)
      
      Note: More than one command can be issued at a time - for example: Holding shift while moving to move with stealth. This action (moving stealthily) 
      must be learned by the agent as opposed to being an option to select from.
  Reward:
      Reward is 1 for every step taken, including the termination step
  Starting State:
      All observations are assigned a uniform random value in [-0.05..0.05]
  Episode Termination:
      The agent dies and is presented with the death screen.

      Solved Requirements
      High score on hard mode
  """

  metadata = {
    'render.modes': ['human'],
    'video.frames_per_second': 120
  }
  
  def __init__(self, agent, screenshot_path):
    self.agent = agent
    self.screenshot_path = screenshot_path 
    self.action_space = spaces.MultiDiscrete(self.agent.action_spaces)
    self.observation_space = None
    self.seed()
    self.viewer = None
    self.state = None
    self.steps_beyond_done = None
    self.screenshot_history = None


  def set_agent(self, agent):
    self.agent = agent


  def get_agent(self):
    return self.agent


  def set_screenshot_history_path(self, history):
    self.screenshot_history = history


  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]


  def step(self, action):
    assert self.screenshot_path != None, "%r (%s) invalid. Be sure to set the location of where the Minecraft client stores screenshots by calling `set_screenshot_paths` when initializing."
    assert self.screenshot_history != None, "%r (%s) invalid. Be sure to set the location of where the environment stores screenshot history by calling `set_screenshot_paths` when initializing."
    state = self.state
    
    threads = []
    for index, action_item in enumerate(action):
      action_thread = self.agent.perform_action(self.agent.actions[index][action_item])
      threads.append(action_thread)
    
    #for thread in threads:
    #  thread.join()
    
    self.get_state()
    done = self.agent.is_dead(self.state)
    
    reward = 0
    if not done:
      reward = reward + 1
    elif self.steps_beyond_done is None:
      # Agent just died!
      self.steps_beyond_done = 0
      reward = 1
    else:
      if self.steps_beyond_done == 0:
        logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
      self.steps_beyond_done += 1
      reward = 0.0

    return self.state, reward, done, {}


  def render(self, mode='human'):
    if self.viewer is None:
      from gym.envs.classic_control import rendering
      self.viewer = rendering.SimpleImageViewer(maxwidth=self.agent.resolution[0])
    if self.state is None:
      state = np.array(Image.new('RGBA',(self.agent.resolution[0],self.agent.resolution[1]), (255, 255, 255, 255)))
    else:
      im = None
      wait_count = 0
      while im is None:
        try:
          im = Image.open(self.state)
        except:
          time.sleep(0.01)
          wait_count = wait_count + 1
          if(wait_count > 100):
            break
      state = np.array(im)  
    return state
    

  def get_state(self):
    self.agent.look()
    files = os.listdir(self.screenshot_path) #images = self.poll_for_screenshot(self.screenshot_path)
    if(len(files) > 0):
      files.sort(reverse=True)
      self.state = os.path.join(self.screenshot_path, files[0])
    return self.state 


  def reset(self):
    self.steps_beyond_done = None    
    if(self.state is not None):
      self.agent.respawn()
    return self.get_state()

