# SculptDiff: Learning Robotic Clay Sculpting from Humans with Goal Conditioned Diffusion Policy
[[arXiv]](https://arxiv.org/abs/2403.10401) [[Project Website]](https://sites.google.com/andrew.cmu.edu/imitation-sculpting/home) [Demonstration Dataset](https://drive.google.com/file/d/1QN1vTGvsCvwakqlCOFcbvscUXZoP7eVo/view)

Manipulating deformable objects remains a challenge within robotics due to the difficulties of state estimation, long-horizon planning, and predicting how the object will deform given an interaction. These challenges are the most pronounced with 3D deformable objects. We propose SculptDiff, a goal-conditioned imitation learning framework that works with point cloud state observations to directly learn clay sculpting policies for a variety of target shapes. To the best of our knowledge this is the first real-world method that successfully learns manipulation policies for 3D deformable objects.

## Download Dataset
Follow the link to the [Demonstration Dataset](https://drive.google.com/file/d/1QN1vTGvsCvwakqlCOFcbvscUXZoP7eVo/view), download and unzip the files and update the path in dataset.py

## Setup PointBERT
Follow the installation instructions and download the model weights of [Point-BERT](https://github.com/Julie-tang00/Point-BERT)

## Train Policies
To train a point cloud-based sculpting policy, run train_policy.py. 

## Replicate Harware Setup
Follow the link to the [Hardware CAD](https://drive.google.com/file/d/1JbbHU8lW7LBvTGYZ2qOLVUAGpOO5gcLK/view?usp=drive_link) to replicate our camera cage.