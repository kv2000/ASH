# ASH: Animatable Gaussian Splats for Efficient and Photoreal Human Rendering (CVPR 2024)

  <p align="center">
    <strong>Haokai Pang&dagger;</strong>
    ·    
    <a href="https://people.mpi-inf.mpg.de/~hezhu/"><strong>Heming Zhu&dagger;</strong></a>
    ·
    <a href="https://gvrl.mpi-inf.mpg.de/"><strong>Adam Kortylewski</strong></a>
    ·
    <a href="https://people.mpi-inf.mpg.de/~theobalt/"><strong>Christian Theobalt</strong></a>
    ·
    <a href="https://people.mpi-inf.mpg.de/~mhaberma/"><strong>Marc Habermann&ddagger;</strong></a>
  </p> 
  <p align="center" style="font-size:15px; margin-bottom:-5px !important;"><sup>&dagger;</sup>Joint first authors.</p>
  <p align="center" style="font-size:15px; margin-bottom:-10px !important;"><sup>&ddagger;</sup>Corresponding author.</p>

---

# News
**2024-6-14** The <strong><font color=green>Training Code</font></strong>, and the <strong><font color=red>Data Processing Code</font></strong> is available! :fireworks::fireworks::fireworks:

**2024-3-29** The initial release, i.e., the <strong><font color=green>Demo Code</font></strong> is available. The <strong><font color=red>Training Code</font></strong> is on the way. For more details, pleaase check out <a href="https://vcai.mpi-inf.mpg.de/projects/ash/"><strong>the project page</strong></a>:smiley:.

---

# Installation
### Clone the repo
```bash
git clone git@github.com:kv2000/ASH.git --recursive

cd ./submodules/diff-gaussian-rasterization/
git submodule update --init --recursive
```
### Install the dependencies

The code is tested on ```Python 3.9```, ```pytorch 1.12.1```, and ```cuda 11.3```.


#### Setup DeepCharacters Pytorch

Firstly, install the underlying clothed human body model, :fireworks:<a href="https://github.com/kv2000/DeepCharacters_Pytorch"><strong>DeepCharacters Pytorch</strong></a>:fireworks:, which also consists the dependencies that needed for this repo.

#### Setup 3DGS

Then, setup the submodules for <a href="https://github.com/graphdeco-inria/gaussian-splatting"><strong>3D Gaussian Splatting</strong></a>.

```bash
# the env with DeepCharacters Pytorch
conda activate mpiiddc 

# 3DGS go
cd ./submodules/diff-gaussian-rasterization/
python setup.py install

cd ../simple-knn/
python setup.py install
```

---
## Setup the metadata and checkpoints
You may find the metadata and the checkpoints <a href="https://gvv-assets.mpi-inf.mpg.de/ASH/"><strong>from this link</strong></a>. 

The extracted metadata and checkpoints follows folder structure below

```bash
# for the checkpoints
checkpoints
|--- Subject0001
    |---deformable_character_checkpoint.pth # character checkpoints
    |---gaussian_checkpoints.tar            # gaussian checkpoints

# for the meta data
meta_data
|--- Subject0001
    |---skeletoolToGTPose                   # training poses
    |   |--- ... 
    |
    |---skeletoolToGTPoseTest               # Testing poses
    |   |--- ...
    |
    |---skeletoolToGTPoseRetarget           # Retartget another subject's pose
    |   |--- ...
    |
    |--- ...                                # Others

```

---
# Run the demo 
Run the following and the results will be stored in ```./dump_results/``` by default.

```bash
bash run_inference.sh
```

---
# Train your model
## Step 1. Data Processing
- Download the compressed  raw data from <a href="https://gvv-assets.mpi-inf.mpg.de/ASH/"><strong>from this link</strong></a> in to ```./raw_data/``` .
- Decompress the data with ```tar -xzvf Subject0022.tar.gz```
- Run the <strong>(slurm) bash script</strong> ```./process_video/bash_get_image.sh``` that extracts the <strong>masked images</strong> from the <strong>raw RGB videoes</strong> and the <strong>foreground mask videoes </strong>. The provided script supports parallel the image extraction with slurm job arrays.

## Step 2. Start Training
Run the following and the results will be stored in ```./dump_results/``` by default.

```bash
bash run_train.sh
```

The folder structure for the training is as follows:

```bash
# for the meta data
dump_results
|--- Subject0022
    |---cached_files                                # The precomputed character related
    |   |--- cached_fin_rotation_quad.pkl
    |   |--- cached_fin_translation_quad.pkl
    |   |--- cached_joints.pkl
    |   |--- cached_ret_canonical_delta.pkl
    |   |--- cached_ret_posed_delta.pkl
    |   |--- cached_temp_vert_normal.pkl
    |
    |---checkpoints                               
    |   |--- ...
    |
    |---exp_stats                                   # Tensorboard Logs
    |   |--- ...
    |
    |---validations_fine                            # Validationed images every X Frames
```

Note that at the <strong>first time</strong> that the training script runs, it will pre-compute and store the <strong>character related data</strong>, stored in ```./dump_results/[Subject Name]/cached_files/```. Which will greatly speed up and reduce the gpu usage of the training process.

## Step 3. Train with your own data.

Plese check out <a href="https://github.com/kv2000/ASH/issues/4"><strong>this issue</strong></a> on some hints on training on your own data, discussion is welcomed :).


---
# Todo list

- [x] Data processing for Training
- [x] Training Code

---

# Citation

If you find our work useful for your research, please, please, please consider citing our paper!

```
@InProceedings{Pang_2024_CVPR,
    author    = {Pang, Haokai and Zhu, Heming and Kortylewski, Adam and Theobalt, Christian and Habermann, Marc},
    title     = {ASH: Animatable Gaussian Splats for Efficient and Photoreal Human Rendering},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {1165-1175}
}
```

---

# Contact
For questions, clarifications, feel free to get in touch with:  
Heming Zhu: hezhu@mpi-inf.mpg.de  
Marc Habermann: mhaberma@mpi-inf.mpg.de  

---
# License
Deep Characters Pyotrch is under [CC-BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) license. The license applies to the pre-trained models and the metadata as well.

---
# Acknowledgements
Christian Theobalt was supported by ERC Consolidator Grant 4DReply (No.770784). Adam Kortylewski was supported by the German Science Foundation (No.468670075). This project was also supported by the Saarbrucken Research Center for Visual Computing, Interaction, and AI. We would also like to thank Andrea Boscolo Camiletto and Muhammad Hamza Mughal for the efforts/discussion on motion retargeting.

Below are some resources that we benefit from (keep updating):

- <a href="https://github.com/graphdeco-inria/gaussian-splatting">3D gaussian-splatting</a>, bravo for the brilliant representation for real-time and high-quality rendering.
- <a href="https://github.com/kv2000/DeepCharacters_Pytorch">DeepCharacters Pytorch</a> for the human character model.
- <a href="https://pytorch3d.org/">Pytorch3D</a> and <a href="https://github.com/kornia/kornia">Kornia</a> for the handy geometry library.
- <a href="https://github.com/Totoro97/NeuS">NeuS</a> for the project structure D:.
