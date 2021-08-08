import argparse

import torch
from torchvision import utils
import cv2
from tqdm import tqdm
import numpy as np

from model import Generator
from config import GANConfig

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--n_img", type=int, default=16, help="number of images to be generated"
    )
    parser.add_argument(
        "--n_row", type=int, default=4, help="number of samples per row"
    )
    parser.add_argument(
        "--truncation", type=float, default=0.5, help="truncation ratio"
    )
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument("--n_frame", type=int, default=120)
    parser.add_argument("--radius", type=float, default=30)
    parser.add_argument(
        "ckpt", metavar="CKPT", type=str, help="path to the model checkpoint"
    )

    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    conf = GANConfig(**ckpt["conf"])
    generator = conf.generator.make().to(device)
    generator.load_state_dict(ckpt["g_ema"], strict=False)
    generator.eval()

    mean_latent = generator.mean_latent(args.truncation_mean)
    x = torch.randn(args.n_img, conf.generator["style_dim"], device=device)

    theta = np.radians(np.linspace(0, 360, args.n_frame))
    x_2 = np.cos(theta) * args.radius
    y_2 = np.sin(theta) * args.radius

    trans_x = x_2.tolist()
    trans_y = y_2.tolist()

    images = []

    transform_p = generator.get_transform(
        x, truncation=args.truncation, truncation_latent=mean_latent
    )

    with torch.no_grad():
        for i, (t_x, t_y) in enumerate(tqdm(zip(trans_x, trans_y), total=args.n_frame)):
            transform_p[:, 2] = t_y
            transform_p[:, 3] = t_x

            img = generator(
                x,
                truncation=args.truncation,
                truncation_latent=mean_latent,
                transform=transform_p,
            )
            images.append(
                utils.make_grid(
                    img.cpu(), normalize=True, nrow=args.n_row, value_range=(-1, 1)
                )
                .mul(255)
                .permute(1, 2, 0)
                .numpy()
                .astype(np.uint8)
            )

    videodims = (images[0].shape[1], images[0].shape[0])
    fourcc = cv2.VideoWriter_fourcc(*"VP90")
    video = cv2.VideoWriter("sample.webm", fourcc, 24, videodims)

    for i in tqdm(images):
        video.write(cv2.cvtColor(i, cv2.COLOR_RGB2BGR))

    video.release()
