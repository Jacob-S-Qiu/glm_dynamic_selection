# Limitations and Enhancements in Genomic Language Models 🧬
##### Open Source Code, Data and models🧾
This repository provides the source code, datasets, and pre-trained models used in our research to advance genomic language modeling. It also includes detailed instructions on how to replicate our experiments, download data and models, and explore additional functionalities. For more details, refer to our preprint on bioRxiv: [Limitations and Enhancements in Genomic Language Models: Dynamic Selection Approach](https://www.biorxiv.org/content/10.1101/2024.11.25.624002v1)

## Quich Start  🚀
### 1. Clone the repository
```bash
git clone https://github.com/Jacob-S-Qiu/glm_dynamic_selection.git
cd glm_dynamic_selection
```
### 2. Install dependencies
Make sure Python 3.9+ is installed, and use the following command to install required packages:
```bash
pip install -r requirements.txt
```
### 3. Download MEME-Suite
Please follow the [official guide](https://meme-suite.org/meme/meme_5.5.6/doc/install.html?man_type=web)

### 4. Usage
- **All experimental files, including data files and model weight files**, are hosted on [Google Drive](https://drive.google.com/drive/folders/1tX7eobxMzt2fH2RZM7mxxmqnDmkR0ulb?usp=sharing). You can download individual files or directly download the **entire project** from Google Drive, which includes the entire repository with code, data, and weights.

- **To train a model**, ensure you have `accelerate` installed and run the following commands:
  ```bash
  pip install accelerate>=0.26.0
  accelerate launch train.py
  ```
  
- **To download the configurations for the three models mentioned in the paper**, use the following links:
  - Hyena: [LongSafari/hyenadna-medium-160k-seqlen-hf](https://huggingface.co/LongSafari/hyenadna-medium-160k-seqlen-hf)
  - NTv2: [InstaDeepAI/nucleotide-transformer-v2-500m-multi-species](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species)
  - CD-GPT: [TencentAI4S/CD-GPT](https://github.com/TencentAI4S/CD-GPT)
 
After the environment is set up, you can start using the repository for various tasks such as data processing, training, or model evaluation.

## Citing Our Work 📝
```plaintext
@article{qiu2024genomic,
  title={Limitations and Enhancements in Genomic Language Models: Dynamic Selection Approach},
  author={Shibo Qiu},
  journal={bioRxiv},
  year={2024},
  doi={10.1101/2024.11.25.624002}
}
```
## License 📚
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## directory structure 🏠
```plaintext
├── all_models_weights 
│   ├── cd_long_base_best_model.pth
│   ├── cd_long_dynamic_best_model.pth
│   ├── cd_long_err_base_best_model.pth
│   ├── cd_long_err_dynamic_best_model.pth
│   ├── cd_short_base_best_model.pth
│   ├── cd_short_dynamic_best_model.pth
│   ├── hy_long_base_best_model.bin
│   ├── hy_long_dynamic_best_model.bin
│   ├── hy_long_err_base_bast_model.bin
│   ├── hy_long_err_dynamic_best_model.bin
│   ├── hy_short_base_best_model.bin
│   ├── hy_short_dynamic_best_model.bin
│   ├── nt_long_base_best_model.safetensors
│   ├── nt_long_dynamic_best_model.safetensors
│   ├── nt_long_err_base_best_model.safetensors
│   ├── nt_long_err_dynamic_best_model.safetensors
│   ├── nt_short_base_best_model.safetensors
│   └── nt_short_dynamic_best_model.safetensors
├── data_long_sequence # Long sequence of the whole process of the experiment
│   ├── data-analysis
│   │   ├── 3model_result # sequence data
│   │   │   ├── 3model_all_correct.fasta
│   │   │   ├── 3model_all_correct.pt
│   │   │   ├── 3model_all_wrong.fasta
│   │   │   ├── 3model_all_wrong.pt
│   │   │   ├── 3model_have_1_correct.fasta
│   │   │   ├── 3model_have_1_correct.pt
│   │   │   ├── 3model_have_2_correct.fasta
│   │   │   └── 3model_have_2_correct.pt
│   │   ├── 3model_result_compare # download and just open, you could view the sequence analysis page
│   │   │   ├── 3model_1_correct_feature_MAST.htm
│   │   │   ├── 3model_1_correct_feature_MEME.htm
│   │   │   ├── 3model_2_correct_feature_MAST.htm
│   │   │   ├── 3model_2_correct_feature_MEME.htm
│   │   │   ├── 3model_all_correct_feature_MAST.htm
│   │   │   ├── 3model_all_correct_feature_MEME.html
│   │   │   ├── 3model_all_wrong_feature_MAST.htm
│   │   │   └── 3model_all_wrong_feature_MEME.htm
│   │   ├── data # all data are available
│   │   │   ├── CDgpt_2_8754.pt
│   │   │   ├── Hyena_602.pt
│   │   │   └── NTv2_240.pt
│   │   ├── step1_similiar.ipynb
│   │   └── step2_feature_analysis.ipynb
│   ├── test.pt
│   ├── train.pt
│   └── valid.pt
├── experiment_long_sequence
│   ├── base_models
│   │   ├── cdgpt
│   │   │   ├── result
│   │   │   │   ├── cdgpt_14590_test_results.pt
│   │   │   └── train&valid&eval
│   │   │       ├── cd_e1_step1_train_on_dataset.py
│   │   │       ├── cd_e1_step2_get_best_model.py
│   │   │       └── cd_e1_step3_get_features_from_best_model.py
│   │   ├── hyena
│   │   │   ├── result
│   │   │   │   ├── hyena_1053_test_feature_final.pt
│   │   │   └── train&valid&eval
│   │   │       ├── hy_e1_step1_train.py
│   │   │       ├── hy_e1_step2_get_best_model.py
│   │   │       └── hy_e1_step3_get_features.py
│   │   ├── ntv2
│   │   │   ├── result
│   │   │   │   ├── ntv2_240_test_results.pt
│   │   │   └── train&valid&eval
│   │   │       ├── nt_e1_step1_tokenize_dataset.py
│   │   │       ├── nt_e1_step2_train.py
│   │   │       ├── nt_e1_step3_get_best_model.py
│   │   │       └── nt_e1_step4_get_features.py
│   │   ├── step5_3model_result
│   │   │   ├── 3model_all_correct.fasta
│   │   │   ├── 3model_all_correct.pt
│   │   │   ├── 3model_all_wrong.fasta
│   │   │   ├── 3model_all_wrong.pt
│   │   │   ├── 3model_have_1_correct.fasta
│   │   │   ├── 3model_have_1_correct.pt
│   │   │   ├── 3model_have_2_correct.fasta
│   │   │   └── 3model_have_2_correct.pt
│   │   ├── step5_3model_result_similiar_compare.ipynb # Compare the differences between the models
│   │   ├── step6_3model_confidence_analysis.ipynb # Analyze the confidence of the model
│   │   ├── step6_merged_file
│   │   │   ├── merged_model_test_data_results.pt
│   │   │   ├── merged_model_train_data_results.pt 
│   │   │   └── merged_model_valid_data_results.pt
│   │   ├── step7_3model_result_fasta
│   │   │   ├── 3model_all_correct.fasta
│   │   │   ├── 3model_all_wrong.fasta
│   │   │   ├── 3model_have_1_correct.fasta
│   │   │   └── 3model_have_2_correct.fasta
│   │   ├── step7_3model_sequence_make_fasta.ipynb # We need to use MEME, so we need a.fasta file
│   │   ├── step8_make_soft_labels_dataset.ipynb 
│   │   └── step8_soft_labels_dataset
│   │       ├── merged_soft_label_and_models_prediction_test_dataset.pt
│   │       ├── merged_soft_label_and_models_prediction_train_dataset.pt 
│   │       ├── merged_soft_label_and_models_prediction_valid_dataset.pt
│   │       ├── soft_labels_test_dataset.pt
│   │       ├── soft_labels_train_dataset.pt 
│   │       └── soft_labels_valid_dataset.pt
│   ├── data
│   │   ├── test.pt
│   │   ├── train.pt 
│   │   └── valid.pt
│   └── dynamic_models
│       ├── cdgpt
│       │   ├── result
│       │   │   └── C7_e1_9000_result_feature.pt
│       │   └── train&valid
│       │       ├── cd_e2_step1_train&get_best_model.py
│       │       └── cd_e2_step2_get_features.py
│       ├── hyena
│       │   ├── result
│       │   │   └── C7_hyena_e1_540_result_feature.pt
│       │   └── train&valid
│       │       ├── hy_e2_step1_train.py
│       │       ├── hy_e2_step2_get_best_model.py
│       │       └── hy_e2_step3_get_features.py
│       ├── ntv2
│       │   ├── result
│       │   │   └── C7_ntv2_e1_380_result_feature.pt
│       │   └── train&valid
│       │       ├── nt_e2_get_best_model.py
│       │       ├── nt_e2_step1_train.py
│       │       └── nt_e2_step3_get_features.py
│       ├── step9
│       │   ├── e2-cdgpt-result
│       │   │   ├── cd_0_right_file.pt
│       │   │   ├── .....(omit 14 files)
│       │   │   └── nt_1_wrong_file.pt
│       │   ├── e2-hyena-result
│       │   │   ├── cd_0_right_file.pt
│       │   │   ├── ......(omit 14 files)
│       │   │   └── nt_1_wrong_file.pt
│       │   └── e2-ntv2-result
│       │       ├── cd_0_right_file.pt
│       │       ├── ......(omit 14 files)
│       │       └── nt_1_wrong_file.pt
│       └── step9-result-data-processing.ipynb
├── experiment_long_sequence_err
│   ├── base_models
│   │   ├── cdgpt
│   │   │   ├── result
│   │   │   │   ├── cdgpt_c10_3036_test_results.pt
│   │   │   └── train&valid&eval
│   │   │       ├── cd_e1_step1_train_on_dataset.py
│   │   │       ├── cd_e1_step2_get_best_model.py
│   │   │       └── cd_e1_step3_get_features_from_best_model.py
│   │   ├── hyena
│   │   │   ├── result
│   │   │   │   ├── hyena_c10_690_test.pt
│   │   │   └── train&valid&eval
│   │   │       ├── hy_e1_step1_train.py
│   │   │       ├── hy_e1_step2_get_best_model.py
│   │   │       └── hy_e1_step3_get_features.py
│   │   ├── ntv2
│   │   │   ├── result
│   │   │   │   ├── ntv2_c10_test_results.pt
│   │   │   └── train&valid&eval
│   │   │       ├── nt_e1_step1_tokenize_dataset.py
│   │   │       ├── nt_e1_step2_train.py
│   │   │       ├── nt_e1_step3_get_best_model.py
│   │   │       └── nt_e1_step4_get_features.py
│   │   ├── step5_3model_result
│   │   │   ├── 2model_all_correct_nt_cd.fasta
│   │   │   ├── 2model_all_correct_nt_cd.htm
│   │   │   ├── 2model_all_correct_nt_cd.pt
│   │   │   ├── 2model_all_wrong_nt_cd.fasta
│   │   │   ├── 2model_all_wrong_nt_cd.htm
│   │   │   ├── 2model_all_wrong_nt_cd.pt
│   │   │   ├── 3model_all_correct.pt
│   │   │   └── 3model_all_wrong.pt
│   │   ├── step5_3model_result_similiar_compare.ipynb
│   │   ├── step6_3model_confidence_analysis.ipynb
│   │   ├── step6_merged_file
│   │   │   ├── merged_model_test_data.pt
│   │   │   ├── merged_model_train_data.pt
│   │   │   └── merged_model_valid_data.pt
│   │   ├── step7_3model_result_fasta
│   │   │   ├── 2model_all_correct_nt_cd.fasta
│   │   │   └── 2model_all_wrong_nt_cd.fasta
│   │   ├── step7_3model_sequence_make_fasta.ipynb
│   │   ├── step8_make_soft_labels_dataset.ipynb
│   │   └── step8_soft_labels_dataset
│   │       ├── merged_soft_label_and_models_prediction_test_dataset.pt
│   │       ├── merged_soft_label_and_models_prediction_train_dataset.pt
│   │       ├── merged_soft_label_and_models_prediction_valid_dataset.pt
│   │       ├── soft_labels_test_dataset.pt
│   │       ├── soft_labels_train_dataset.pt
│   │       └── soft_labels_valid_dataset.pt
│   ├── data
│   │   ├── C10_cdgpt_1kbp_test_dataset.pt
│   │   ├── C10_cdgpt_1kbp_train_dataset.pt
│   │   ├── C10_cdgpt_1kbp_valid_dataset.pt
│   │   ├── C10_hyena_20kbp_test_dataset.pt
│   │   ├── C10_hyena_20kbp_train_dataset.pt
│   │   ├── C10_hyena_20kbp_valid_dataset.pt
│   │   ├── C10_ntv2_12kbp_test_dataset.pt
│   │   ├── C10_ntv2_12kbp_train_dataset.pt# this file is in google drive(too big)
│   │   ├── C10_ntv2_12kbp_valid_dataset.pt
│   │   └── step1_get_DNA_from_Ensemble.ipynb
│   └── dynamic_models
│       ├── cdgpt
│       │   ├── result
│       │   │   └── C11_cdgpt_e2_10200_result_feature.pt
│       │   └── train&valid
│       │       ├── cd_e2_step1_train&get_best_model.py
│       │       └── cd_e2_step2_get_features.py
│       ├── hyena
│       │   ├── result
│       │   │   └── C11_hyena_e2_800_result_feature.pt
│       │   └── train&valid
│       │       ├── hy_e2_step1_train.py
│       │       ├── hy_e2_step2_get_best_model.py
│       │       └── hy_e2_step3_get_features.py
│       ├── ntv2
│       │   ├── result
│       │   │   └── C7_ntv2_e1_380_result_feature.pt
│       │   └── train&valid
│       │       ├── nt_e2_get_best_model.py
│       │       ├── nt_e2_step1_train.py
│       │       └── nt_e2_step3_get_features.py
│       ├── step9
│       │   ├── cdgpt-e2-result-data-analysis
│       │   │   ├── cd_0_right_file.pt
│       │   │   ├── ......(omit 14 files)
│       │   │   └── nt_1_wrong_file.pt
│       │   ├── hyena-e2-result-data-analysis
│       │   │   ├── cd_0_right_file.pt
│       │   │   ├── ......(omit 14 files)
│       │   │   └── nt_1_wrong_file.pt
│       │   └── ntv2-e2-result-data-analysis
│       │       ├── cd_0_right_file.pt
│       │       ├── ......(omit 14 files)
│       │       └── nt_1_wrong_sequences.fasta
│       └── step9-result-data-processing.ipynb
├── experiment_short_sequence
│   ├── base_models
│   │   ├── cdgpt
│   │   │   ├── result
│   │   │   │   ├── cdgpt_c12_human_enhancers_cohn_test_10420_results.pt
│   │   │   └── train&valid&eval
│   │   │       ├── cd_e1_step1_train_on_dataset.py
│   │   │       ├── cd_e1_step2_get_best_model.py
│   │   │       └── cd_e1_step3_get_features_from_best_model.py
│   │   ├── hyena
│   │   │   ├── result
│   │   │   │   ├── hyena_c12_e1_human_enhancers_cohn_test_120.pt
│   │   │   └── train&valid&eval
│   │   │       ├── hy_e1_step1_train.py
│   │   │       ├── hy_e1_step2_get_best_model.py
│   │   │       └── hy_e1_step3_get_features.py
│   │   ├── ntv2
│   │   │   ├── result
│   │   │   │   ├── ntv2_c12_human-enhancers-cohn_test_results_200.pt
│   │   │   └── train&valid&eval
│   │   │       ├── nt_e1_step1_tokenize_dataset.py
│   │   │       ├── nt_e1_step2_train.py
│   │   │       ├── nt_e1_step3_get_best_model.py
│   │   │       └── nt_e1_step4_get_features.py
│   │   ├── step5_3model_result
│   │   │   ├── 3model_all_correct.pt
│   │   │   ├── 3model_all_wrong.pt
│   │   │   ├── 3model_have_1_correct.pt
│   │   │   └── 3model_have_2_correct.pt
│   │   ├── step5_3model_result_similiar_compare.ipynb
│   │   ├── step6_3model_confidence_analysis.ipynb
│   │   ├── step6_merged_file
│   │   │   ├── merged_model_test_data.pt
│   │   │   └── merged_model_train_data.pt
│   │   ├── step7_3model_result_fasta
│   │   │   ├── 3model_all_correct.fasta
│   │   │   ├── 3model_all_wrong.fasta
│   │   │   ├── 3model_have_1_correct.fasta
│   │   │   └── 3model_have_2_correct.fasta
│   │   ├── step7_3model_sequence_make_fasta.ipynb
│   │   ├── step8_make_soft_labels_dataset.ipynb
│   │   └── step8_soft_labels_dataset
│   │       ├── merged_soft_label_and_models_prediction_test_dataset.pt
│   │       ├── merged_soft_label_and_models_prediction_train_dataset.pt
│   │       ├── step1_soft_labels_test_dataset.pt
│   │       └── step1_soft_labels_train_dataset.pt
│   ├── data
│   │   ├── dataset_summary.csv
│   │   ├── genomic_benchmark_datasets # benchmarks
│   │   │   ├── test_demo_coding_vs_intergenomic_seqs.pt
│   │   │   ├── test_demo_human_or_worm.pt
│   │   │   ├── test_drosophila_enhancers_stark.pt
│   │   │   ├── test_human_enhancers_cohn.pt
│   │   │   ├── test_human_enhancers_ensembl.pt
│   │   │   ├── test_human_ensembl_regulatory.pt
│   │   │   ├── test_human_nontata_promoters.pt
│   │   │   ├── test_human_ocr_ensembl.pt
│   │   │   ├── train_demo_coding_vs_intergenomic_seqs.pt
│   │   │   ├── train_demo_human_or_worm.pt
│   │   │   ├── train_drosophila_enhancers_stark.pt
│   │   │   ├── train_human_enhancers_cohn.pt
│   │   │   ├── train_human_enhancers_ensembl.pt
│   │   │   ├── train_human_ensembl_regulatory.pt
│   │   │   ├── train_human_nontata_promoters.pt
│   │   │   └── train_human_ocr_ensembl.pt
│   │   └── test.ipynb
│   └── dynamic_models
│       ├── cdgpt
│       │   ├── result
│       │   │   └── C12_e2_1_6000_result_feature.pt
│       │   └── train&valid
│       │       ├── cd_e2_step1_train&get_best_model.py
│       │       └── cd_e2_step2_get_features.py
│       ├── hyena
│       │   ├── result
│       │   │   └── C12_hyena_e2_140_result_feature.pt
│       │   └── train&valid
│       │       ├── hy_e2_step1_train.py
│       │       ├── hy_e2_step2_get_best_model.py
│       │       └── hy_e2_step3_get_features.py
│       ├── ntv2
│       │   ├── result
│       │   │   └── C12_e2_ntv2_120_result_feature.pt
│       │   └── train&valid
│       │       ├── nt_e2_get_best_model.py
│       │       ├── nt_e2_step1_train.py
│       │       └── nt_e2_step3_get_features.py
│       ├── step9
│       │   ├── cdgpt-e2-result-data-analysis
│       │   │   ├── cd_0_right_file.pt
│       │   │   ├── ......(omit 14 files)
│       │   │   └── nt_1_wrong_file.pt
│       │   ├── hyena-e2-result-data-analysis
│       │   │   ├── cd_0_right_file.pt
│       │   │   ├── ......(omit 14 files)
│       │   │   └── nt_1_wrong_file.pt
│       │   └── ntv2-e2-result-data-analysis
│       │       ├── cd_0_right_file.pt
│       │       ├── ......(omit 14 files)
│       │       └── nt_1_wrong_sequences.fasta
│       └── step9-result-data-processing.ipynb
└── readme.md
```
