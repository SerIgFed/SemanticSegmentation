import argparse

import torch
from semantic_segmentation import models
from semantic_segmentation import load_model


def to_onnx(input_path, output_path, height, width):
    batch_size = 2
    model = load_model(models['BiSeNetV2'], torch.load(input_path))
    model.eval().cuda()
    x = torch.ones((batch_size, 3, height, width)).cuda()
    input_names = ["input"]
    output_names = ["output", "aux_c2", "aux_c3", "aux_c4", "aux_c5"]  # from bisenetv2.py:310
    torch.onnx.export(model, x, output_path, verbose=True, input_names=input_names,
                      output_names=output_names, opset_version=11)


# width must be divisible by 32?
# height must be divisible by 8?
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=864)
    parser.add_argument('--input_path', default='models/model_segmentation_person2_30.pt')
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    to_onnx(args.input_path, args.output_path, args.height, args.width)
