# Robot Motion Control with LoRa Tuned Pre-Trained Vision-Language-Action
The project develops a system for robotic movement with reinforcement learning for low-level motion control and OpenVLA for high-level navigation, enabling autonomous movement and path planning.

# Robot Motion Control Training
Everything can be ran from the AutoNAV.ipynb jupyter notebook
1. Run AutoNAV.ipynb, select p.GUI to run with the PyBullet GUI or p.DIRECT for headless version
2. Results can be visualized using results_vis.ipynb

# OpenVLA Custom Fine-Tune
1. The "Create Dataset" notebook creates our data from our simulation environment in pybullet. We will have both text commands and simulation images for the LoRa fine-tuning.
2. Clone the RLDS github(https://github.com/kpertsch/rlds_dataset_builder), put the RLDS_Custom notebook in the home directory. This is how you port our custom simulation environment dataset to the format for the OpenVLA fine-tuning script. This will result in a tfds data format to be passed in. Use custom_dataset_dataset_builder in a custom_dataset folder, which also needs a __init__.py to run the commandn !tfds build --datasets=custom_dataset.
3. First clone OpenVLA (https://github.com/openvla/openvla/tree/main), then put this notebook in the home directory. The OpenVLA_run notebook demonstrates how to run the fine-tuning script onces the data has been converted into the RLDs format with tfds. This will save the fine-tuned model to "openvla/runs/" in the example we have shown.

