# Material Prediction for Design Automation Using Graph Representation Learning

## Repository Structure
```
.
├─ feature_extraction       # Helper functions for extracting features
│  ├─ material_categories
│  └─ MVCNN                 # MVCNN code for generating visual embeddings
│      └─ helpers
│ 
├─ fully_algorithm          # Experiment No.1: Fully Algorithm-guided Prediction
│  ├─ GNN.py                # GNN implementation
│  ├─ config.py             # Hyperparameter and experiment configuration tuning knobs
│  ├─ dataloader.py         # Generation of graphs and dataloaders
│  ├─ process_data.py       # Preprocessing of input data
│  └─ train.py              # Training of the model (implementation of the experiment)
│ 
├─ partial_algorithm        # Experiment No.2: Partial Algorithm-guided Prediction
├─ sample_data              # A tiny subset of data sampled from the a03.10 Fusion 360 Assembly Gallery
└─ user_guided              # Experiment No.3: User-guided Prediction
```

## Dataset
You may obtain the latest dataset (the one used in the manuscript) [here](https://drive.google.com/file/d/106rpRc3G7SYQt6crJJvr3Src5Kt_UJvc/view?usp=sharing)
