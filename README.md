# ASH: Animatable Gaussian Splats for Efficient and Photoreal Human Rendering (CVPR 2024)

  <p align="center">
    <strong>Haokai Pang*</strong>
    ·    
    <a href="https://people.mpi-inf.mpg.de/~hezhu/"><strong>Heming Zhu*</strong></a>
    ·
    <a href="https://gvrl.mpi-inf.mpg.de/"><strong>Adam Kortylewski</strong></a>
    ·
    <a href="https://people.mpi-inf.mpg.de/~theobalt/"><strong>Christian Theobalt</strong></a>
    ·
    <a href="https://people.mpi-inf.mpg.de/~mhaberma/"><strong>Marc habermann#</strong></a>
  </p> 

---

# News
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
--- Subject0001
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
# Todo list

- [ ] Data processing and training
- [ ] ...


---
# Contributers

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
