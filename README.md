# Recursive Self-Enhancing Reinforcement Learning (RSE-RL)
Code for reproducing results in **Recursive Self-Improvement for Camera Image and Signal Processing**.

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


