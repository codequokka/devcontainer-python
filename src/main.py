from __future__ import annotations

import cv2
import numpy as np
from numpy import uint8
from numpy.typing import NDArray


def match_pattern(image_path: str, template_path: str, result_path: str) -> None:
    image: NDArray[uint8] = cv2.imread(image_path)
    template: NDArray[uint8] = cv2.imread(template_path)

    result: NDArray[uint8] = cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    print(min_val, max_val, min_loc, max_loc)

    threshold = 0.999
    location = np.where(result >= threshold)

    w, h, _ = template.shape

    for pt in zip(*location[::-1]):
        print(pt)
        cv2.rectangle(image, pt, (pt[0] + w, pt[-1] + h), (0, 0, 255), 1)

    cv2.imwrite(result_path, image)


if __name__ == "__main__":
    # match_pattern(
    #     "images/inputs/images/mesoko.jpg",
    #     "images/inputs/templates/cm_logo.jpg",
    #     "images/outputs/result.jpg",
    # )
    match_pattern(
        image_path="images/inputs/images/mario_field.png",
        template_path="images/inputs/templates/mario_coin.png",
        result_path="images/outputs/result.jpg",
    )
