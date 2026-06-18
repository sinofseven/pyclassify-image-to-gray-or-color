from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def save_image(tmp_path: Path) -> Callable[..., Path]:
    """numpy 配列を PNG(可逆)として tmp_path に保存しパスを返すヘルパ。

    JPEG はクロマサブサンプリング/非可逆圧縮で Cb/Cr が変化し決定論性が
    崩れるため、必ず PNG を使う。
    """

    def _save(array: np.ndarray, name: str = "img.png", mode: str = "RGB") -> Path:
        path = tmp_path / name
        Image.fromarray(array.astype(np.uint8), mode=mode).save(path)
        return path

    return _save


def _make_gray_array(height: int = 100, width: int = 100, value: int = 128) -> np.ndarray:
    """R=G=B の一様グレー画像配列を作る。"""
    return np.full((height, width, 3), value, dtype=np.uint8)


def _make_color_array(
    height: int = 100,
    width: int = 100,
    background: int = 128,
    patch: tuple[int, int, int] = (220, 30, 30),
    patch_size: int = 10,
) -> np.ndarray:
    """ニュートラルなグレー背景に小さな彩度パッチを置いたカラー画像配列を作る。

    背景がグレーなので Cb/Cr の中央値は 128 付近に保たれ、パッチ画素のみ
    chroma が大きく逸脱して color として検出される。
    """
    arr = _make_gray_array(height, width, background)
    arr[0:patch_size, 0:patch_size] = patch
    return arr


@pytest.fixture
def make_gray_array() -> Callable[..., np.ndarray]:
    """一様グレー画像配列を作るビルダを返す fixture。"""
    return _make_gray_array


@pytest.fixture
def make_color_array() -> Callable[..., np.ndarray]:
    """グレー背景+彩度パッチのカラー画像配列を作るビルダを返す fixture。"""
    return _make_color_array
