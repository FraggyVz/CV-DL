# Refactor regression fixture

Generated from `00e024f:stratscout_prototype.ipynb` before migration, using
Python 3.13, NumPy 2.2.6, OpenCV 4.12.0 and Torch 2.14.1 CPU.
Contains deterministic synthetic trajectories, frames, features, tracking, heatmaps
and untrained detector logits. These verify behaviour preservation, not accuracy.
Seeds: background 1234, trajectories/rendering 17, detector weights 7.
Configuration: 32 pixel crop, 8 timesteps, default five slots per team.
