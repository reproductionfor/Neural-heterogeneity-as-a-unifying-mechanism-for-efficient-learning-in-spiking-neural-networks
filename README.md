Code Availability

The code is currently being prepared and will be publicly shared before the final publication of this paper.
All experiments are organized by learning methods and can be run directly following the instructions below.

🧠 Running RLS and FORCE Experiments

The RLS- and FORCE-based experiments are implemented in MATLAB.

Each experiment can be executed directly within the MATLAB environment without additional dependencies.

The provided scripts reproduce the reservoir computing and dynamic trajectory fitting results presented in the paper.

⚙️ Running SGD-Based Experiments

The SGD-based deep SNN experiments are implemented in Python (PyTorch).

To reproduce the results, navigate to the SuGD_code directory and run:

python main.py


Configuration files and hyperparameters correspond to those reported in Tables II–IV of the manuscript.

📦 Datasets

All datasets used in this study are publicly available.
Please download them from the official sources listed below:

Heidelberg Spiking Datasets (SHD and SSC)
https://zenkelab.org/resources/spiking-heidelberg-datasets-shd

Neuromorphic-MNIST (N-MNIST)
https://www.garrickorchard.com/datasets/n-mnist

DVS Gesture
https://research.ibm.com/publications/a-low-power-fully-event-based-gesture-recognition-system
