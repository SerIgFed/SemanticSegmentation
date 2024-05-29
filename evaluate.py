import argparse
import logging
import pathlib
import functools
import time
import numpy as np

import cv2
import torch
from torchvision import transforms

from semantic_segmentation import models
from semantic_segmentation import load_model
from semantic_segmentation import draw_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default='GPU',
                        help='Optional. Specify the target device to infer on: CPU or GPU (def)')
    parser.add_argument('--images', type=str, default=None,
                        help='Optional. Specify folder with images')
    parser.add_argument('--video', type=str, default=None)
    parser.add_argument('--rotate', action='store_true',
                        help='Optional. Rotate each given frame by 180 deg. before process')

    parser.add_argument('--out_video', type=str, default=None)

    parser.add_argument('--model', type=str, default='models/model_segmentation_person2_30.pt')
    parser.add_argument('--model-type', type=str, choices=models, default='BiSeNetV2')

    parser.add_argument('--threshold', type=float, default=0.5)

    parser.add_argument('--save', action='store_true')
    parser.add_argument('--display', action='store_true')

    return parser.parse_args()


def find_files(dir_path: pathlib.Path, file_exts):
    assert dir_path.exists()
    assert dir_path.is_dir()

    for file_ext in file_exts:
        yield from dir_path.rglob(f'*{file_ext}')


def _load_image(image_path: pathlib.Path):
    image = cv2.imread(str(image_path))
    assert image is not None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_width  = (image.shape[1] // 32) * 32
    image_height = (image.shape[0] // 32) * 32

    image = image[:image_height, :image_width]
    return image


def process_cv_image(img):
    img = cv2.resize(img, (853, 480))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_width = (img.shape[1] // 32) * 32
    img_height = (img.shape[0] // 32) * 32
    return img[:img_height, :img_width]


def get_color(num):
    colors = np.array([[0, 0, 127], [0, 127, 0], [127, 0, 0], [127, 127, 0], [127, 0, 127], [0, 127, 127],
                       [127, 63, 0], [127, 0, 63], [63, 127, 0], [63, 0, 127], [0, 127, 63], [0, 63, 127]],
                      dtype=np.uint8)
    return colors[num]


def infer_on_video(args, device):
    display = args.display or args.out_video == None

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ratio = video_width / video_height
    target_height = 480
    target_width = int(target_height * ratio)
    image_width  = (target_width // 32) * 32
    image_height = (target_height // 32) * 32
    if not display:
        writer = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (image_width, image_height))
    times = []

    delay = 1
    esc_code = 27
    p_code = 112
    mean_time = 0

    while True:
        current_time = cv2.getTickCount()
        ret, frame = cap.read()
        if not ret:
            break

        if args.rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame = cv2.resize(frame, (target_width, target_height))
        frame = frame[:image_height, :image_width]

        beg = time.process_time()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = frame.astype(np.float16)
        img = torch.from_numpy(frame.transpose((2, 0, 1)))  # permute_channels
        # img = img.type(torch.HalfTensor)
        #img = img.cuda()
        img = img.div(255)
        img = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(img)
        with torch.no_grad():
            img = img.to(device).unsqueeze(0)
            output = model(img)['out']
            output = torch.sigmoid(output)

            output = output > args.threshold
            end = time.process_time()
            times.append(end - beg)
        for ind, (cat, cat_image, mask_image) in enumerate(draw_results(img[0], output[0], categories=model.categories)):
            if np.max(mask_image):
                mask_pixels = np.where(mask_image > 0)[:2]
                frame[mask_pixels] //= 2
                frame[mask_pixels] = frame[mask_pixels] + get_color(ind)
                mask_image[mask_image > 0] = 1
                mask_image[:, :, 0][mask_image[:, :, 1] > 0] = 1
                mask_image[:, :, 0][mask_image[:, :, 2] > 0] = 1
                mask_image = cv2.resize(mask_image, (3840, 2160), cv2.INTER_NEAREST)
                mask_image = mask_image.astype(np.uint8) * 255
        # cat, cat_image, _ = next(draw_results(img[0], output[0], categories=model.categories))
        # writer.write(cat_image)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if display:
            current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
            if mean_time == 0:
                mean_time = current_time
            else:
                mean_time = mean_time * 0.95 + current_time * 0.05
            cv2.putText(frame, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
                        (40, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
            cv2.imshow('Masked frame', frame)
            key = cv2.waitKey(delay)
            if key == esc_code:
                break
            if key == p_code:
                if delay == 1:
                    delay = 0
                else:
                    delay = 1
        else:
            writer.write(frame)

    print(np.mean(times))
    cap.release()
    if not display:
        writer.release()


def infer_on_images_folder(args, device):
    assert args.display or args.save

    logging.info(f'evaluating images from {args.images}')
    if args.rotate:
        logging.info('flag rotate is not supported')
    
    image_dir = pathlib.Path(args.images)

    fn_image_transform = transforms.Compose(
        [
            transforms.Lambda(lambda image_path: _load_image(image_path)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    times = []
    for image_file in find_files(image_dir, ['.png', '.jpg', '.jpeg']):
        logging.info(f'segmenting {image_file} with threshold of {args.threshold}')

        image = fn_image_transform(image_file)

        with torch.no_grad():
            begin = time.process_time()
            image = image.to(device).unsqueeze(0)
            results = model(image)['out']
            results = torch.sigmoid(results)

            results = results > args.threshold
            end = time.process_time()
            times.append(end - begin)

        for category, category_image, mask_image in draw_results(image[0], results[0], categories=model.categories):
            if args.save:
                output_name = f'results_{category}_{image_file.name}'
                logging.info(f'writing output to {output_name}')
                cv2.imwrite(str(output_name), category_image)
                cv2.imwrite(f'mask_{category}_{image_file.name}', mask_image)

            if args.display:
                cv2.imshow(category, category_image)
                cv2.imshow(f'mask_{category}', mask_image)

        if args.display:
            if cv2.waitKey(0) == ord('q'):
                logging.info('exiting...')
                exit()
    print(np.mean(times))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    if args.video == None and args.images == None:
        raise ValueError('Either --video or --image has to be provided')

    if args.device == 'GPU' and not torch.cuda.is_available():
        print('No CUDA device found, inferring on CPU')
    device = 'cuda:0' if torch.cuda.is_available() and args.device == 'GPU' else 'cpu'
    logging.info(f'running inference on {device}')

    logging.info(f'loading {args.model_type} from {args.model}')
    model = torch.load(args.model, map_location=device)
    model = load_model(models[args.model_type], model)
    # model.half()
    model.to(device).eval()

    if args.images is None:
        infer_on_video(args, device)
    else:
        infer_on_images_folder(args, device)
