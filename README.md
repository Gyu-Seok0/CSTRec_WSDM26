# Capturing User Interests from Data Streams for Continual Sequential Recommendation

This repository provides the source code of our paper: ["Capturing User Interests from Data Streams for Continual Sequential Recommendation (CSTRec)"](https://www.arxiv.org/abs/2506.07466), accepted at WSDM 2026.

<p align="center">
  <img src="images/method_fix.png" alt="CSTRec Method Overview" width="70%">
</p>

---

## 1. Environment Setup

```bash
# Create the Conda environment
conda env create -f env.yml

# Activate the environment
conda activate CSTRec
```

---

## 2. Usage

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

## 3. Citation
If you use this code or our method in your research, please cite our paper:

```bibtex
@article{lee2025leveraging,
  title={Leveraging Historical and Current Interests for Continual Sequential Recommendation},
  author={Lee, Gyuseok and Yoo, Hyunsik and Hwang, Junyoung and Kang, SeongKu and Yu, Hwanjo},
  journal={arXiv preprint arXiv:2506.07466},
  year={2025}
}
```
