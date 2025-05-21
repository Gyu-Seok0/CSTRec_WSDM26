# Overview

This repository contains the source code for our paper: **Capturing User Interests from Data Streams for Continual Sequential Recommendation (CSTRec)**

<p align="center">
  <img src="images/method_fix.png" alt="CSTRec Method Overview" width="70%">
</p>

---

# Environment Setup

```bash
# Create the Conda environment
conda env create -f env.yml

# Activate the environment
conda activate CSTRec
```

---

# Usage

1. **Unzip the dataset**

   ```bash
   unzip <dataset>.zip
   ```
2. **Run the model**
   ```bash
   ./run.sh [dataset]
   ```
* Supported datasets: gowalla, ml-1m, yelp

    Note: The run.sh script includes a --fc (fast check) option by default for quick testing.
To execute the full training and evaluation pipeline, remove the --fc flag.