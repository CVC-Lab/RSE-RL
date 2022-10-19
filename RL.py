import gym
from gym import spaces
from stable_baselines3 import A2C
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from evaluation import *
from cv2 import imshow

class Img_Enhancing_Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, clear_images, noise_images, models, target_score, display_result=True):
        super(Img_Enhancing_Env, self).__init__()
        self.clear_images = clear_images
        self.noise_images = noise_images
        self.test_images = np.array(clear_images)
        self.target_score = target_score
        self.display_result = display_result

        self.init_weights = []
        for model in models:
            self.init_weights.append(model.transform.trainable_variables)
        self.models = models
        self.reward_range = (-15, 5)
        self.psnr = 0

        self.h = len(clear_images[0][0])
        self.current_step = 0
        self.action_space = spaces.Box(low=0.999, high=1.001, shape=(LATENT_DIM*3,), dtype=np.float32)
        self.observation_space = spaces.Box(0, self.target_score, shape=(1,),dtype=np.float32)
    
    def step(self, action):
        for k in range(len(self.init_weights)):
            for i in range(len(self.init_weights[k])):
                self.models[k].transform.trainable_variables[i].assign(tf.multiply(self.init_weights[k][i], 
                                                                       action[k*LATENT_DIM: (k+1)*LATENT_DIM]))
        
        clear_ys, clear_us, clear_vs = cvt_bgr_yuv(self.clear_images)
        noise_ys, noise_us, noise_vs = cvt_bgr_yuv(self.noise_images)
        
        clear_ys, noise_ys = gen_train_set(clear_ys, noise_ys, SHAPE, BLOCK_SIZE, NUM_BLOCK, OVERLAP)
        clear_us, noise_us = gen_train_set(clear_us, noise_us, SHAPE, BLOCK_SIZE, NUM_BLOCK, OVERLAP)
        clear_vs, noise_vs = gen_train_set(clear_vs, noise_vs, SHAPE, BLOCK_SIZE, NUM_BLOCK, OVERLAP)
        
        recons_ys = reconstruct_image(noise_ys, model_y, BATCH_SIZE, BLOCK_PER_IMAGE, WIDTH, HEIGHT, BLOCK_SIZE, OVERLAP)
        recons_us = reconstruct_image(noise_us, model_u, BATCH_SIZE, BLOCK_PER_IMAGE, WIDTH, HEIGHT, BLOCK_SIZE, OVERLAP)
        recons_vs = reconstruct_image(noise_vs, model_v, BATCH_SIZE, BLOCK_PER_IMAGE, WIDTH, HEIGHT, BLOCK_SIZE, OVERLAP)

        recons_images = cvt_yuv_bgr(recons_ys, recons_us, recons_vs)

        psnr = quality_evaluation(recons_images, self.test_images, metric='PSNR', display=False)
        reward = 1.25 * (psnr - self.target_score) + 5
        reward = self.reward_range[1] if reward > self.reward_range[1] else reward
        reward = self.reward_range[0] if reward < self.reward_range[0] else reward
        done = psnr >= self.target_score - 1
        self.current_step += 1
        self.psnr = psnr
        if self.display_result:
            print(self.current_step, reward, quality_evaluation(clear_images, recons_images))
            imshow(recons_images[0])
        obs = np.array([psnr])
        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        print('reset')
        self.psnr = 0
        self.current_step = 0
        obs = np.array([0])
        return obs

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(f'Step: {self.current_step}')
        print(f'PSNR: {self.psnr}')
        
