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
    center_chroma: float | None = field(default=None)


def classify_gray_or_color(
    image_path: str | Path,
    *,
    max_size: int = 768,
    chroma_threshold: float = 10.0,
    strong_chroma_threshold: float = 25.0,
    color_ratio_threshold: float = 0.003,
    strong_color_ratio_threshold: float = 0.0003,
    center_chroma_threshold: float = 5.0,
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

    # 中央値そのものがニュートラル(128,128)からどれだけ離れているか。
    # 一様な単色画像は中央値減算で chroma が全画素 0 になり「ばらつき」では
    # 検出できないため、この center_chroma をフォールバック判定に使う。
    center_chroma = float(np.sqrt((cb_center - 128) ** 2 + (cr_center - 128) ** 2))

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
    elif color_ratio == 0.0 and center_chroma >= center_chroma_threshold:
        # ばらつきが全く検出されない完全一様画像のみ、中央値がニュートラルから
        # 離れていれば一様な単色とみなして color と判定する。
        is_color = True
        reason = (
            f"uniform color detected: center_chroma {center_chroma:.4f} "
            f">= {center_chroma_threshold}"
        )
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
        center_chroma=center_chroma,
    )
