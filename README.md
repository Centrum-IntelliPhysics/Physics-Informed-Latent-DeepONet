# Physics-Informed-Latent-DeepONet
[Sharmila Karumuri](https://scholar.google.com/citations?user=uY1G-S0AAAAJ&hl=en), [Lori Graham-Brady](https://scholar.google.com/citations?user=xhj8q8cAAAAJ&hl=en) and [Somdatta Goswami](https://scholar.google.com/citations?user=GaKrpSkAAAAJ&hl=en).

You can find both the presentation slides and the recording of our presentation, outlining our approach and the results we achieved:

Slides- [PI_Latent_NOs.pdf](https://github.com/user-attachments/files/20778869/PI_Latent_NOs.pdf)

Recording- [PI_Latent_NOs.mp4](https://github.com/user-attachments/assets/9aaa582b-8629-462f-b59f-b21fe763bccf)

In this work, we introduce a Physics-Informed Latent Neural Operator (PI-Latent-NO) framework that learns mappings between functions in a reduced latent space and reconstructs the solution in the original space in a purely physics-informed manner.

Traditional physics-informed deeponets (PI-Vanilla-NO) often require heavily overparameterized architectures, leading to longer training times and convergence difficulties. To address these challenges, we propose PI-Latent-NO, an architecture featuring two coupled DeepONets trained end-to-end:
	•	A Latent-DeepONet that learns a compact, low-dimensional representation of the solution, and
	•	A Reconstruction-DeepONet that maps the latent representation back to the physical space.

By embedding PDE constraints into training via automatic differentiation, our approach eliminates the need for labeled data and ensures physics-consistent predictions. The proposed framework is both memory and compute-efficient, exhibiting near-constant scaling with problem size and delivering significant speedups over traditional physics-informed operator models.

We validate our method on a range of high-dimensional parametric PDEs, demonstrating its accuracy, scalability, and potential for real-time inference in complex physical systems.

<p align="center">
  <img src="https://github.com/user-attachments/assets/6e1cb7b3-3831-42c6-b623-24eb89467312" 
       alt="Schematic of the physics-informed vanilla neural operator (PI-Vanilla-NO)" 
       width="600"/>
</p>
<p align="center"><b>Figure 1:</b> Schematic of the physics-informed vanilla neural operator (PI-Vanilla-NO).</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/1c7f2381-730d-40c6-bcd8-a569780cf2c5" 
       alt="Schematic of our Physics-Informed Latent Neural Operator (PI-Latent-NO)" 
       width="600"/>
</p>
<p align="center"><b>Figure 2:</b> Schematic of our Physics-Informed Latent Neural Operator (PI-Latent-NO).</p>

## Results
<p align="center">
  <img width="972" alt="Screenshot 2025-06-06 at 9 10 41 AM" src="https://github.com/user-attachments/assets/4caf0cab-665c-4ad0-bc09-9cdccdc507a8" />
</p>
<p align="center"> Comparison of the PI-Vanilla-NO and the PI-Latent-NO results for a 2D Stove-Burner  Simulation: (a) runtime per iteration (seconds/iteration), and (b) memory (MB).
</p>

## Data
The labeled dataset used for ablation studies and for comparing our model's predictions with the ground truth in our paper can be found [here](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/sgoswam4_jh_edu/Eqzr4ur3WpNIo8zT5gjuosgBWtZ8kD6ldRl-yx2741kF6A?e=2DDiqr).

## Installing
The code for examples is written in PyTorch. To clone our repository:
```
git clone https://Centrum-IntelliPhysics/Physics-Informed-Latent-DeepONet.git
cd Physics-Informed-Latent-DeepONet
```
## Repository Overview
This repository contains implementations and analyses for the examples described in the paper. It is organized as follows:
* utils folder: Under this we have the key files required for forward-mode automatic differentiation, neural network architectures for the respective frameworks, as well as scripts for generating and visualizing plots.
* Examples folder:  Each example discussed in the paper is organized into its own folder. Within these folders, you will find Jupyter notebooks titled 'a_Vanilla-NO.ipynb' and 'b_Latent-NO.ipynb'.  Additionally, comparison studies of memory usage and training runtime for two of the examples are provided in separate folders, with names starting with 'Comparison_studies'."
* Results: Results from the computations are saved as jupyter notebooks in the respective 'results' folder of each example.  
  
### Citation:
If you use this code for your research, please cite our paper [http://arxiv.org/abs/2409.13280](https://arxiv.org/pdf/2501.08428).
