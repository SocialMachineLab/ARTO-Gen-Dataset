# Experiments

These experiments informed three design decisions in building ARTO-KG: the artistic styles, the image generation model and its prompt configuration, and the object detectors used for validation. The images and results of the generation and detection experiments are in the `experiments/` folder of [youngcan1/ARTO-Gen-Dataset](https://huggingface.co/datasets/youngcan1/ARTO-Gen-Dataset) on Hugging Face.

## Artistic Style Selection

`style_feature_analysis/` contains the analysis behind the choice of the five ARTO-KG styles from 12 WikiArt styles. Each candidate is assessed on four criteria: separability from the other styles in feature space, zero-shot style classification accuracy, object detection performance on its generated and real images, and feature similarity between its generated and real images.

The scripts use WikiArt images in one folder per style, with the folder names listed in `config.py`; the image location is set with `STYLE_IMAGE_DIR`. They also use single-object images generated in each style, stored under `outputs/validation_images/`. Run the scripts from this folder, starting with `wikiart_extract_features.py`, which samples the images and extracts the features used by the other scripts. Then run the other `wikiart_*.py` and `generated_*.py` scripts, and finally the `analyze_*.py` scripts, which summarise the results. All results are written to `outputs/`.

## Image Generation Model and Prompt Configuration

`multimodel_generation/` selects the image generation model: SD-XL-base, FLUX.1-dev, SD-3.5-large, and Qwen-Image each generate images for the 70 ARTO-KG specifications in `experiment_dataset/`, which cover the five styles at three levels of scene complexity. Each model receives the prompt format matched to its text-conditioning characteristics: keywords for SD-XL-base, a single sentence for SD-3.5-large, and a detailed description for FLUX.1-dev and Qwen-Image. The generated images are evaluated by the ARTO-KG validation pipeline, and the resulting scores are used to compare the models.

From this folder, generate images with each model (`qwen`, `flux`, `sd35`, or `sdxl`) and validate them in turn:

```bash
pip install -r requirements.txt
python src/generate.py --model flux --dataset-dir experiment_dataset \
    --output-dir experiment_results/black-forest-labs_FLUX.1-dev
python src/validate.py --model flux
```

## Object Detector Selection

`object_detection/` tests YOLOv11-large, RF-DETR-large, Grounding DINO, and OWLv2 on 500 ARTO-KG artworks with manually annotated bounding boxes, 100 per style, to choose the detectors used for validation. The Hugging Face folder provides the images and ground-truth annotations, together with saved predictions that can be scored without running the detectors:

```bash
python od_evaluation.py --data-dir /path/to/experiments/object_detection --from-saved
```

Without `--from-saved`, the detectors run on the images instead. Predictions are matched to ground-truth boxes with the same object label at an IoU threshold of 0.5. The script reports mAP@0.5 and micro-averaged precision and recall for each style and overall, and writes them to `evaluation_results.csv` in the data folder.
