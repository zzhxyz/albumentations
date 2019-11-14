import io
import cv2
import argparse

import numpy as np
import matplotlib.pyplot as plt


def load_rgb_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_figure(height=1080, width=1920, dpi=200):
    return plt.figure(figsize=(width / dpi, height / dpi))


def figure_to_numpy(figure, dpi=200):
    buf = io.BytesIO()
    figure.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def apply_transform(
    image, transform_cls, args, targets=None, nrows=None, ncols=None, height=1080, width=1920, dpi=200
):

    args = [{"p": 0}] + list(args)
    targets = targets or {}
    total = len(args)

    if nrows is None:
        nrows = total // np.sqrt(total).astype(int)
    if ncols is None:
        ncols = total // nrows

    if nrows * ncols != total:
        ncols += 1

    result_args = []
    figure = get_figure(height, width, dpi)
    figure.tight_layout(pad=0)
    for i, args_i in enumerate(args, 1):
        if "p" not in args_i:
            args_i["p"] = 1

        if i == 1:
            title = "Original"
        else:
            title = ", ".join([f"{name}={val}" for name, val in args_i.items()])

        ax = figure.add_subplot(nrows, ncols, i)
        ax.set_title(title)
        ax.axis("off")
        ax.imshow(transform_cls(**args_i)(image=image, **targets)["image"])

        result_args.append(args_i)

    figure.canvas.draw()
    plt.axis("off")
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.1, top=0.9, hspace=0.2, wspace=0.01)
    result_image = figure_to_numpy(figure, dpi)
    plt.close(figure)

    return result_args, result_image


def create_docs_text(cls, args):
    name = cls.__name__
    module = cls.__module__

    text = name + "\n" + "-" * len(name) + "\n\n"
    text += f"API link: :class:`~{module}`\n\n"

    for i, args_i in enumerate(args, 1):
        if args_i["p"] == 0:
            text += f"{i}. Original image"
            continue
        args_i = ", ".join(f"{key}={value}" for key, value in args_i.items())
        text += f"{i}. :code:`{name}({args_i})`\n"

    text += f"\n.. figure:: ./images/{name}.jpg\n    :alt: {name} image"

    return text


def parse_args():
    parser = argparse.ArgumentParser(description="Create text for transform example")
    parser.add_argument("-a", "--arg", action="append", help="transform argument name")
    parser.add_argument("-v", "value", action="append", help="transform argument value")
    parser.add_argument(
        "-s", "--save", metavar="DIR", default=None, help="path to docs file to write results at end of it is."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    # image = load_rgb_image('e:/Macaw Parrots image.jpg')
    #
    # args, image = apply_transform(
    #     image,
    #     A.Blur,
    #     [
    #         {'blur_limit': [7, 7]},
    #         {'blur_limit': [15, 15]},
    #         {'blur_limit': [30, 30]},
    #         {'blur_limit': [60, 60]}
    #     ]
    # )
    #
    # cv2.imshow("", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # cv2.waitKey(0)
