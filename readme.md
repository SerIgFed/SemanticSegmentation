# Semantic Segmentation
[Source](https://github.com/WillBrennan/SemanticSegmentation)
## Usage
This repository contains code for semantic segmentation using `BiSeNetV2` or `FCNResNet101`.
The pretrained `BiSeNetV2` model for people segmentation located under models/model_segmentation_person2_30.pt.

You can convert it to the `onnx` format using the `to_onnx.py` script to use it the pipeline.

You can also run video segmentation with `.pt` model and train a new model using COCO dataset.

### Install
Optionally create virtual environment:
```
python -m venv venv
source venv/bin/activate (on Linux) or
.\venv\Scripts\activate (on Windows)
```
Install requirements:
```
pip install -r requirements.txt
```
### Convert pre-trained model to .onnx
```
python to_onnx.py --height <out_height> --width <out_width> --input_path models/model_segmentation_person2_30.pt --output_path <onnx_out_path>
```
### Required width, height and output_path
Given that full_frame_width has size `\[full_width, full_height\]`,
"tracker:sem_seg_scale" is set to `sem_seg_scale` in config.

You need create ONNX model with
* height >= full_height / sem_seg_scale, height % 32 == 0, height is the minimum possible such value 
* width >= height * full_width / full_height, width % 32 == 0, width is the minimum possible such value
* output_path = \<path to Pipeline\>/people_segm/models/segmentation/bisenet_\<width\>x\<height\>.onnx

For example:
Full frame width is 3840x2160, sem_seg_scale is 4.5. Then
* height = 480
* width = 864
* output_path is \<path to Pipeline\>/people_segm/models/segmentation/bisenet_864x480.onnx

Another example:
Full frame width is 3120x2340, sem_seg_scale is 4.5. Then
* height = 544
* width = 736
* output_path is \<path to Pipeline\>/people_segm/models/segmentation/bisenet_736x544.onnx

### Video segmentation
```
python evaluate_images.py --in_video <input video> --out_video <output video> --model models/model_segmentation_person2_30.pt --model-type BiSeNetV2
```

### Train model
Download `2017 Train images` and `2017 Stuff Train/Val annotations` from the [COCO Dataset](https://cocodataset.org/#download).

Locate it under `datasets/COCO`

Extract images and annotations for `person` class:

```
python extract_from_coco.py --images datasets/COCO/val2017/ --annotations datasets/COCO/annotations/instances_val2017.json --output datasets/person_coco_images --categories person 
```

Check dataset markup:

```
python check_dataset.py --dataset datasets/person_coco_images_train
```

Run training:
```
python train.py --train datasets/person_coco_images_train --val datasets/person_coco_images_val --model-tag segmentation_person --model-type BiSeNetV2
```

For more information, check readme_source.md.