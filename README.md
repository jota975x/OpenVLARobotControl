# Robot Motion Control with LoRa Tuned Pre-Trained Vision-Language-Action
This project explores the integration of video and text input for autonomous robot navigation using OpenVLA (Vision and Language Agent) and a reinforcement learning-based robot motion control system. The OpenVLA model combines visual and textual data to generate robot commands, which are then passed to a separate module responsible for controlling the robot's linear and angular velocities. The OpenVLA model is trained using Low-Rank Adaptation (LoRA) techniques, while the motion control module leverages the Soft Actor-Critic (SAC) algorithm to optimize robot performance. The objective of this work is to develop an end-to-end system capable of autonomous navigation, where the robot interprets and responds to multi-modal input to navigate in dynamic environments.

# Robot Motion Control Training
Everything can be ran from the AutoNAV.ipynb jupyter notebook
1. Run AutoNAV.ipynb, select p.GUI to run with the PyBullet GUI or p.DIRECT for headless version
2. Results can be visualized using results_vis.ipynb

State, Goal and Reward defintion can be observed in `utils.py`, additionally, this file includes functions for differential driving and environment set up. Model definition is located in `SAC.py`. LiDAR data collection function is located in `lidar.py`. Robot URDF files and meshes can be obtained from the 'diff_drive_robot' folder.

# OpenVLA Custom Fine-Tune
1. The "Create Dataset" notebook creates our data from our simulation environment in pybullet. We will have both text commands and simulation images for the LoRa fine-tuning.
2. Clone the RLDS github(https://github.com/kpertsch/rlds_dataset_builder), put the RLDS_Custom notebook in the home directory. This is how you port our custom simulation environment dataset to the format for the OpenVLA fine-tuning script. This will result in a tfds data format to be passed in. Use custom_dataset_dataset_builder in a custom_dataset folder, which also needs a __init__.py to run the commandn !tfds build --datasets=custom_dataset.
3. First clone OpenVLA (https://github.com/openvla/openvla/tree/main), then put this notebook in the home directory. The OpenVLA_run notebook demonstrates how to run the fine-tuning script onces the data has been converted into the RLDs format with tfds. This will save the fine-tuned model to "openvla/runs/" in the example we have shown.

