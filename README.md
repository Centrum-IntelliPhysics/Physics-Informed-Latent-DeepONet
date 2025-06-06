# Physics-Informed-Latent-DeepONet

In this work, we introduce a Physics-Informed Latent Neural Operator (PI-Latent-NO) framework that learns mappings between functions in a reduced latent space and reconstructs the solution in the original space in a purely physics-informed manner.

Traditional physics-informed deeponets (PI-Vanilla-NO) often require heavily overparameterized architectures, leading to longer training times and convergence difficulties. To address these challenges, we propose PI-Latent-NO, an architecture featuring two coupled DeepONets trained end-to-end:
	•	A Latent-DeepONet that learns a compact, low-dimensional representation of the solution, and
	•	A Reconstruction-DeepONet that maps the latent representation back to the physical space.

By embedding PDE constraints into training via automatic differentiation, our approach eliminates the need for labeled data and ensures physics-consistent predictions. The proposed framework is both memory and compute-efficient, exhibiting near-constant scaling with problem size and delivering significant speedups over traditional physics-informed operator models.

We validate our method on a range of high-dimensional parametric PDEs, demonstrating its accuracy, scalability, and potential for real-time inference in complex physical systems.

![Schematic of the physics-informed vanilla neural operator (PI-Vanilla-NO)](https://github.com/user-attachments/files/20628450/Schematic-PIVanillaNO.pdf)
![Schematic of our Physics-Informed Latent Neural Operator (PI-Latent-NO)](https://github.com/user-attachments/files/20628453/Schematic-PILatentNO.pdf)
 
