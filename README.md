# Recursive Self-Enhancing Reinforcement Learning (RSE-RL)

[[paper]](https://www.semanticscholar.org/paper/Reinforcement-Learning-of-Self-Enhancing-Camera-and-Bajaj-Wang/5755a54efe2b4744b62d050bd6569e1b401bc710) 

Code for reproducing results in **Reinforcement Learning of Self Enhancing Camera Image and Signal Processing**.

Our RSE-RL model views the identification and correction of artifacts as a recursive self-learning and self-improvement exercise and consists of two major sub-modules: (i) The latent feature sub-space clustering/grouping obtained through an equivariant variational auto-encoder enabling rapid identification of the correspondence and discrepancy between noisy and clean image patches. (ii) The adaptive learned transformation controlled by a trust-region soft actor-critic agent that progressively filters and enhances the noisy patches using its closest feature distance neighbors of clean patches. Artificial artifacts that may be introduced in a patch-based ISP, are also removed through a reward-based de-blocking recovery and image enhancement.  We demonstrate the self-improvement feature of our model by recursively training and testing on images, wherein the enhanced images resulting from each epoch provide a natural data augmentation and robustness to the RSE-RL training-filtering pipeline.

![atchetecture](https://github.com/CVC-Lab/rse_rl/blob/main/figures/struct.png)

## Requirements:
See requirement.txt\
Run
`pip install -r requirement.txt` \
GPU is required

## Datasets:
- `CelebA` - 4GB. CelebA-HQ 256x256 dataset. Downloaded from [here](https://openaipublic.azureedge.net/glow-demo/data/celeba-tfr.tar)
- `SIDD-Medium Dataset` - 12GB. Smartphone Image Denoising Dataset consists of 320 image pairs (noisy and ground-truth). Download from [here](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php)

Please put the decompressed datasets in the same directory with the code during experiments, otherwise please set **cwd** in utils.py to the file directory where the datasets locate at.

## Experiments:
#### An small scale experiment example can be found in the [notebook](https://github.com/yunhaoyang234/RSE-RL/blob/main/RSE_RL.ipynb)

#### CelebA Denoising Experiment
```bash
$ python3 experiment_celeba.py \
          --file_batch 5\
          --epoch 50\
          --latent_dim 72\
          --target_psnr 30\
    	    --train_files_path "celeba_train(REPLACE THIS WITH YOUR OWN FILE DIRECTORY)/"\
    	    --test_files_path "celeba_test(REPLACE THIS WITH YOUR OWN FILE DIRECTORY)/"
```
The reconstruction quality of test data evaluated by PSNR, SSIM, and UQI will be printed out, and the trained model will be saved to the `pretrained model` folder. The PSNR for the self-enhancing reinforcement learning will be printed out.

Note: train_files_path and test_files_path should be the directory (folder) that contains all the tfrecord files.

#### SIDD Denoising Experiment
```bash
$ python3 experiment_sidd_denoise.py \
          --file_batch 5\
          --epoch 25\
          --latent_dim 96\
          --target_psnr 34\
          --train_files_path "sidd_noise(REPLACE THIS WITH YOUR OWN FILE DIRECTORY)/"\
          --validation_file_path "sidd_ground_truth(REPLACE THIS WITH YOUR OWN FILE DIRECTORY)/"\
          --test_files_path "sidd_test_noise(REPLACE THIS WITH YOUR OWN FILE DIRECTORY)/"\
          --test_validation_files_path "sidd_test_GT(REPLACE THIS WITH YOUR OWN FILE DIRECTORY)/"
    	  
```
The denoisng quality of the SIDD Benchmark images evaluated by PSNR, SSIM, and UQI will be printed out, and the trained model will be saved to the `pretrained model` folder. The PSNR for the self-enhancing reinforcement learning will be printed out.

## Visualizations:
#### CelebA Denoising Results
Synthesized Noisy Images:
![noisy](https://github.com/yunhaoyang234/RSE-RL/blob/main/figures/noise_i.png)
RSE-RL Denoising Results:
![denoise](https://github.com/yunhaoyang234/RSE-RL/blob/main/figures/denoise.png)

#### SIDD Denoising Results
![sidd1](https://github.com/yunhaoyang234/RSE-RL/blob/main/figures/sidd2.png)
![sidd2](https://github.com/yunhaoyang234/RSE-RL/blob/main/figures/sidd3.png)

#### Reinforcement Learning Recursive Enhancing Results
![recursive](https://github.com/yunhaoyang234/RSE-RL/blob/main/figures/recursive_dif.png)

## Acknowledgements

This project is a collective result of Yunhao Yang, Yuhan Zheng, Yi Wang, and Dr. Chandrajit Bajaj

## Cite us

```bash
@inproceedings{Bajaj2021ReinforcementLO,
  title={Reinforcement Learning of Self Enhancing Camera Image and Signal Processing},
  author={Chandrajit Bajaj and Yi Wang and Yunhao Yang},
  year={2022}
}
```
