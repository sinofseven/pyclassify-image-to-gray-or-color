import dataclasses

import numpy as np
import pytest
from PIL import Image

from pyclassify_image_to_gray_or_color import (
    ClassifyResult,
    classify_gray_or_color,
)


def test_pure_gray_image_is_classified_as_gray(save_image, make_gray_array):
    """R=G=B の一様グレー画像は color と判定されない。"""
    path = save_image(make_gray_array(value=128))

    result = classify_gray_or_color(path)

    assert result.is_color is False
    assert result.color_ratio == 0.0
    assert result.strong_color_ratio == 0.0
    # valid 画素があるので中央値が float で返る(≈128)
    assert isinstance(result.cb_center, float)
    assert isinstance(result.cr_center, float)
    assert result.cb_center == pytest.approx(128.0, abs=1.0)
    assert result.cr_center == pytest.approx(128.0, abs=1.0)


def test_gray_background_with_color_patch_is_classified_as_color(
    save_image, make_color_array
):
    """グレー背景に小さな彩度パッチを置いた画像は color と判定される。"""
    path = save_image(make_color_array())

    result = classify_gray_or_color(path)

    assert result.is_color is True
    assert result.reason.startswith("color detected")
    assert result.color_ratio > 0.0
    assert result.strong_color_ratio > 0.0


def test_uniform_color_fill_is_not_color_due_to_median_subtraction(
    save_image, make_gray_array
):
    """一様に塗りつぶした色は中央値減算で chroma が 0 になり color にならない。"""
    # 全面を彩度の高い赤で塗る(R=G=B ではないが一様)
    arr = make_gray_array()
    arr[:, :] = (220, 30, 30)
    path = save_image(arr)

    result = classify_gray_or_color(path)

    assert result.is_color is False
    assert result.color_ratio == 0.0


def test_all_black_image_returns_too_few_valid_pixels(save_image, make_gray_array):
    """全黒画像は valid 画素がなく専用の分岐を通る。"""
    path = save_image(make_gray_array(value=0))

    result = classify_gray_or_color(path)

    assert result.is_color is False
    assert result.reason == "valid pixels are too few"
    assert result.color_ratio == 0.0
    assert result.strong_color_ratio == 0.0
    assert result.cb_center is None
    assert result.cr_center is None


def test_all_white_image_returns_too_few_valid_pixels(save_image, make_gray_array):
    """全白画像も y < 248 を満たさず valid 画素ゼロになる。"""
    path = save_image(make_gray_array(value=255))

    result = classify_gray_or_color(path)

    assert result.is_color is False
    assert result.reason == "valid pixels are too few"
    assert result.cb_center is None
    assert result.cr_center is None


def test_rgba_opaque_color_patch_is_classified_as_color(save_image, make_color_array):
    """不透明(alpha=255)の RGBA 画像でもクラッシュせず color 判定できる。"""
    rgb = make_color_array()
    rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
    path = save_image(rgba, name="rgba.png", mode="RGBA")

    result = classify_gray_or_color(path)

    assert result.is_color is True


def test_rgba_fully_transparent_patch_is_composited_onto_white(
    save_image, make_color_array
):
    """完全透明(alpha=0)の彩度パッチは白背景に合成され color に寄与しない。"""
    rgb = make_color_array()
    alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    # 彩度パッチ領域だけ完全透明にする
    alpha[0:10, 0:10] = 0
    rgba = np.dstack([rgb, alpha])
    path = save_image(rgba, name="rgba_transparent.png", mode="RGBA")

    result = classify_gray_or_color(path)

    # 透明パッチは白(255,255,255)に置き換わり、残りはグレー → color にならない
    assert result.is_color is False


def test_la_mode_grayscale_is_classified_as_gray(save_image):
    """LA(グレースケール+alpha)モードの画像は gray と判定される。"""
    gray = np.full((100, 100), 128, dtype=np.uint8)
    alpha = np.full((100, 100), 255, dtype=np.uint8)
    la = np.dstack([gray, alpha])
    path = save_image(la, name="la.png", mode="LA")

    result = classify_gray_or_color(path)

    assert result.is_color is False


@pytest.mark.parametrize("as_str", [True, False])
def test_accepts_str_and_path(save_image, make_color_array, as_str):
    """image_path に str と Path の双方を渡せて結果が一致する。"""
    path = save_image(make_color_array())
    arg = str(path) if as_str else path

    result = classify_gray_or_color(arg)

    assert result.is_color is True


def test_high_thresholds_flip_color_to_gray(save_image, make_color_array):
    """閾値を極端に大きくすると同じカラー画像でも gray 判定になる。"""
    path = save_image(make_color_array())

    assert classify_gray_or_color(path).is_color is True

    result = classify_gray_or_color(
        path,
        color_ratio_threshold=1.0,
        strong_color_ratio_threshold=1.0,
    )

    assert result.is_color is False
    assert result.reason.startswith("color_ratio")


def test_classify_result_is_frozen():
    """ClassifyResult は frozen のため属性代入できない。"""
    result = ClassifyResult(
        is_color=False,
        reason="x",
        color_ratio=0.0,
        strong_color_ratio=0.0,
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.is_color = True


def test_exif_orientation_image_is_handled_without_error(
    save_image, make_color_array, tmp_path
):
    """EXIF orientation タグ付き画像でも例外なく判定できる。"""
    img = Image.fromarray(make_color_array(height=80, width=120), mode="RGB")
    exif = img.getexif()
    exif[0x0112] = 6  # Orientation = 90 度回転
    path = tmp_path / "exif.png"
    img.save(path, exif=exif)

    result = classify_gray_or_color(path)

    assert isinstance(result, ClassifyResult)
    assert result.is_color is True
