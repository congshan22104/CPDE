# CPDE: Centralized Planning with Decentralized Execution

This repository contains the official implementation of our paper:

> **Centralized Planning with Decentralized Execution for Counter-UAV Operations in Complex Urban Environments**

---
## 📐 Framework

![CPDE Framework](assets/framework.png)
## 📖 Overview
This paper presents a **Centralized Planning with Decentralized Execution (CPDE)** framework for counter-UAV operations in complex urban environments.  

The framework integrates:
- **State Estimation**: robust estimation from noisy/missing sensing data.  
- **Trajectory Prediction**: forecasting intruder UAV motion.  
- **Spatiotemporal Planning**: generating interception anchors at the planning layer.  
- **Decentralized Execution**: defender UAVs use local velocity commands without inter-UAV communication.  

CPDE enables coordinated interception under **imperfect sensing**, **high-speed intruders**, and **communication disruptions**.  

**Experimental results** in a simulated urban environment show that CPDE significantly outperforms existing methods in both **interception rate** and **efficiency**, demonstrating effectiveness in cluttered urban settings.

---

## ⚙️ Requirements
- Python >= 3.8
- [PyBullet](https://pybullet.org/wordpress/) (simulation)
- PyTorch (for learning modules, if applicable)
- NumPy, SciPy, Matplotlib
