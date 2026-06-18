from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class ClassifyResult:
    is_color: bool
    reason: str
    color_ratio: float
    strong_color_ratio: float
    cb_center: float | None = field(default=None)
    cr_center: float | None = field(default=None)


def classify_gray_or_color(
    image_path: str | Path,
    *,
    max_size: int = 768,
    chroma_threshold: float = 10.0,
    strong_chroma_threshold: float = 25.0,
    color_ratio_threshold: float = 0.003,
    strong_color_ratio_threshold: float = 0.0003,
) -> ClassifyResult:
    """
    return:
      ("gray", info)  or  ("color", info)

    gray:
      AVIF yuv400 にしてよさそう

    color:
      AVIF yuv444 / yuv420 などカラー保持した方がよさそう
    """

    path = Path(image_path)

    with Image.open(path) as img:
        # EXIF回転を反映
        img = ImageOps.exif_transpose(img)

        # 透過がある場合は白背景に合成
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            bg.alpha_composite(img.convert("RGBA"))
            img = bg.convert("RGB")
        else:
            img = img.convert("RGB")

        # 高速化のため縮小して判定
        img.thumbnail((max_size, max_size))

        ycbcr = img.convert("YCbCr")
        arr = np.asarray(ycbcr, dtype=np.float32)

    y = arr[:, :, 0]
    cb = arr[:, :, 1]
    cr = arr[:, :, 2]

    # 真っ黒・真っ白に近い部分はノイズで色が乗りやすいので少し除外
    valid = (y > 8) & (y < 248)

    if not np.any(valid):
        return ClassifyResult(
            is_color=False,
            reason="valid pixels are too few",
            color_ratio=0.0,
            strong_color_ratio=0.0,
        )

    # スキャン画像の全体的な色かぶりを無視するため、
    # Cb/Cr の中央値を基準にする
    cb_center = np.median(cb[valid])
    cr_center = np.median(cr[valid])

    chroma = np.sqrt((cb - cb_center) ** 2 + (cr - cr_center) ** 2)

    color_pixels = valid & (chroma > chroma_threshold)
    strong_color_pixels = valid & (chroma > strong_chroma_threshold)

    valid_count = int(np.count_nonzero(valid))
    color_ratio = float(np.count_nonzero(color_pixels) / valid_count)
    strong_color_ratio = float(np.count_nonzero(strong_color_pixels) / valid_count)

    is_color = (
        color_ratio >= color_ratio_threshold
        or strong_color_ratio >= strong_color_ratio_threshold
    )

    if is_color:
        parts = []
        if color_ratio >= color_ratio_threshold:
            parts.append(f"color_ratio {color_ratio:.4f} >= {color_ratio_threshold}")
        if strong_color_ratio >= strong_color_ratio_threshold:
            parts.append(
                f"strong_color_ratio {strong_color_ratio:.4f} >= {strong_color_ratio_threshold}"
            )
        reason = "color detected: " + ", ".join(parts)
    else:
        reason = (
            f"color_ratio {color_ratio:.4f} < {color_ratio_threshold} and "
            f"strong_color_ratio {strong_color_ratio:.4f} < {strong_color_ratio_threshold}"
        )

    return ClassifyResult(
        is_color=is_color,
        reason=reason,
        color_ratio=color_ratio,
        strong_color_ratio=strong_color_ratio,
        cb_center=float(cb_center),
        cr_center=float(cr_center),
    )
