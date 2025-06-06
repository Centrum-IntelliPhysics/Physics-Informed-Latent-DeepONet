# Physics-Informed-Latent-DeepONet

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
<img width="972" alt="Screenshot 2025-06-06 at 9 10 41 AM" src="https://github.com/user-attachments/assets/4caf0cab-665c-4ad0-bc09-9cdccdc507a8" />
**Figure:** Comparison of the PI-Vanilla-NO and the PI-Latent-NO results for a 2D Stove-Burner  Simulation: (a) runtime per iteration (seconds/iteration), and (b) memory (MB).

## Data
The labeled dataset used for ablation studies and for comparing our model's predictions with the ground truth in our paper can be found [here](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/sgoswam4_jh_edu/Eqzr4ur3WpNIo8zT5gjuosgBWtZ8kD6ldRl-yx2741kF6A?e=2DDiqr).

