from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from pyisetcam import AssetStore, camera_compute, camera_create, scene_from_file


def main() -> None:
    asset_store = AssetStore.default()
    rows, cols = 96, 144
    rr, cc = np.indices((rows, cols))
    rgb = np.zeros((rows, cols, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.round(255.0 * cc / max(cols - 1, 1)).astype(np.uint8)
    rgb[:, :, 1] = np.round(255.0 * rr / max(rows - 1, 1)).astype(np.uint8)
    rgb[:, :, 2] = np.round(255.0 * (1.0 - cc / max(cols - 1, 1))).astype(np.uint8)

    scene = scene_from_file(rgb, "rgb", 100.0, "lcdExample.mat", asset_store=asset_store)
    camera = camera_compute(camera_create(asset_store=asset_store), scene, asset_store=asset_store)

    output_dir = Path("reports/parity")
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "scene_from_file_input.png"
    result_path = output_dir / "scene_from_file_result.png"
    iio.imwrite(input_path, rgb)
    iio.imwrite(result_path, np.clip(np.round(camera.fields["ip"].data["srgb"] * 255.0), 0, 255).astype(np.uint8))
    print(input_path)
    print(result_path)


if __name__ == "__main__":
    main()
