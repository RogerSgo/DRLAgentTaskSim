<h1> Task DRL: Trayectory Tracking </h1>

![Zona de trabajo](https://github.com/RogerSgo/DRLAgentTaskSim/blob/main/Informative_Image.png)
<h2> Description </h2>

The Robotic Manipulator executes the task of tracking trajectories located within a work zone of the CoppeliaSim simulation scene using a Deep Reinforcement Learning based control approach.
<h2> Software: </h2>

- CoppeliaSim 4.7 64 bits
- Gymnasium 0.29
- Numpy 1.25
- Matplotlib 3.7.1
- Python 3.11
- Stable Baselines3 2.0
<h2> Contenido </h2>

- CoppeliaSim Scene for Evaluation and Training.
- .ipynb files for Evaluation and Training.
- Trained model with 10000 episodes (best_model.zip).
<h2> Procedure </h2>

- The CoppeliaSIm file Entorno_MRR_DRL_IK.ttt is used for training and inference of the DRL model.
- Train: Open, modify parameters of the file TrainingAgentDRL.ipynb as the user sees fit and run training scene in CoppeliaSim.
- Inference: Open the file InferenciaAgenteDRL.ipynb, load the previously trained model and run to evaluate the behavior of the trained agent.
<h2> Multimedia content </h2>

Youtube link of the model inference: https://www.youtube.com/watch?v=n-YulJUdDHg
