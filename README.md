# Fine-Tuning Geospatial Foundation Models for Land Cover Classification

This repository contains the code and experimental logs for fine-tuning the **TerraMind** foundation model on the **LUCAS (Land Use and Coverage Area frame Survey)** dataset. The project evaluates classification capabilities on a small ($n \approx 2,000$), highly imbalanced dataset of multispectral satellite imagery chips from Italy.

## Project Overview
- **Objective:** Classify 10 distinct land cover categories (e.g., Arable Lands, Wooden Areas, Water Bodies) using multispectral inputs.
- **Model:** TerraMind (Base and Tiny variants).
- **Challenge:** Extreme class imbalance.
- **Core Experiment:** Comparing loss functions (Class-Weighted Cross-Entropy vs. Focal Loss) and decoder architectures (Identity vs. MLP) to handle imbalance.

## Methodology
**Data Pipeline:** Retrieval of Sentinel-2 derived imagery, statistical normalization, and a custom **stratified train-validation split** to preserve rare classes.

**Architecture:**
    * **Backbone:** TerraMind-v1-base (Pre-trained).
    * **Decoders:** Tested **Identity Decoder** (lightweight linear probing) vs. **MLP Decoder** (non-linear spatial processing).

**Training Strategy:**
    * **Optimizer:** AdamW (lr=1e-4) with reduced weight decay (0.05).
    * **Precision:** 16-bit mixed precision for efficiency.
    * **Loss Functions:** Evaluated **Class-Weighted Cross-Entropy** against **Class-Weighted Focal Loss**.

## Key Findings
* **The Trade-off:** There is a critical tension between optimization and fairness.
    * **Focal Loss** achieved excellent convergence (Test Loss $\approx$ 0.49) and high Micro-Accuracy ($\approx$ 54%) by maximizing performance on dominant classes. However, it suffered from **minority class collapse**, failing to recognize rare categories.
    * **Weighted Cross-Entropy** yielded higher loss ($\approx$ 1.69) but superior **Macro Accuracy** ($\approx$ 40%), maintaining significantly better recall for underrepresented classes.
* **Decoder Impact:** The choice between MLP and Identity decoders had negligible impact on performance ($<0.5\%$ difference), suggesting the pre-trained backbone features are already robust.

* Full report available at [Geospatial Foundation Models Report](docs/Geospatial_Foundation_Models_Report.pdf)

## How to Run
1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install git+[https://github.com/IBM/terratorch.git](https://github.com/IBM/terratorch.git)
    ```
2.  **Download Data:**
    The notebook automatically handles data retrieval from Google Drive using `gdown`.
3.  **Run the Notebook:**
    Open `Geospatial_Foundation_Models_PiA_Project_Italy.ipynb` and execute the cells. The training loop uses PyTorch Lightning.

## Requirements
* Python 3.10+
* PyTorch & TorchGeo
* TerraTorch (IBM)
* Albumentations
* PyTorch Lightning

## Author
**Tito Capovilla**
*Geospatial Foundation Models - Passion in Action*
*Politecnico di Milano*
