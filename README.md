# SAE Meets DUSt3R  

## Overview  
This project explores the integration of **Sparse Autoencoders (SAE)** with **DUSt3R**, a 2D-to-3D reconstruction model. Our goal is to improve **interpretability** by analyzing feature representations in DUSt3R’s **Vision Transformer (ViT) encoder and decoder layers**.  

## Installation  

### 1. Clone the Repository  
```bash
git clone https://github.com/VladKozlovskiy/SAEmeetsDUTS3R.git  
cd SAEmeetsDUTS3R
```
2. Install Dependencies
   
```pip install -r requirements.txt```  
Usage

Run training and evaluation scripts:

```
python train_encoder_sae.py  # SAE on ViT encoder layer  
python train_decoder_sae.py  # SAE on decoder layer  
python analyze_results.py    # Visualize activated concepts  
```

Repository Structure

```
📂 SAEmeetsDUTS3R
├── models/         # SAE and DUSt3R model definitions
├── data/           # Datasets
├── results/        # Outputs and visualizations
├── requirements.txt # Dependencies
└── README.md       # Project documentation
```

You can find checkpoints via the link 

We used Arkit & BlendedMVS for training, you shiuld load them on your own in the fromt origin DUST3R processe. 


## Results & Findings

### ✅ Encoder Layer Insights
Multi-view consistency: Concepts activate in the same regions across different views.
Structured representation: The layer behaves like two concatenated ViT encoders, processing both views while preserving alignment.
High-level thematic activation: Features consistently activate for images of the same category.

### 🔴 Decoder Layer Observations
Single-view activation: Unlike the encoder, concepts activate only in one image, suggesting independent processing.
Lack of multi-view consistency: This may indicate that the decoder focuses on local feature reconstruction rather than global 3D alignment.
Further research needed: How can we enforce multi-view consistency in the decoder to improve 3D reconstruction quality?
Discussion & Future Research

- 🧩 Should the decoder be modified to process features across multiple views instead of treating each separately?
- 🎯 How does sparsity influence decoder layer feature specialization?
- 🔄 Can a hybrid sparsity strategy improve both interpretability and reconstruction quality?
- 🏗 Would integrating cross-view attention mechanisms in the decoder lead to better generalization?


## Contributors

- 👨‍💻 Vlad Kozlovsky – Training & GitHub setup
- 📊 Sameer Tantry – Literature review & visualization
- 📈 Sofya Savelyeva – Data processing & results interpretation

For more details, refer to our final report. 🚀
