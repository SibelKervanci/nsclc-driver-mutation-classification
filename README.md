# nsclc-driver-mutation-classification
Gene expression-based three-class LUAD driver mutation classification (EGFR/KRAS/TN) using ComBat, SHAP, and heterogeneous ensemble learning
# NSCLC Driver Mutation Classification

Three-class classification of LUAD driver mutation subtypes 
(EGFR/KRAS/Triple-Negative) using gene expression data.

## Methods
- ComBat batch correction (neuroCombat)
- SHAP-based feature selection (top 500 genes)
- Bayesian hyperparameter optimization (Optuna)
- Heterogeneous ensemble: XGBoost, LightGBM, 
  Random Forest, CatBoost, SVM (SoftVoting)
- External validation: TCGA-LUAD (n=508)

## Data
- Internal: GSE31210, GSE13213, GSE72094 (n=774, GEO)
- External: TCGA-LUAD via UCSC Xena

## Requirements
See requirements.txt

## Citation
[Kervancı IS, Özsert Yiğit G. — manuscript under review]

## License
MIT License
