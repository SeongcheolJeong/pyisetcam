from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from pyisetcam import AssetStore, camera_compute, camera_create, scene_create


def main() -> None:
    asset_store = AssetStore.default()
    scene = scene_create(asset_store=asset_store)
    camera = camera_compute(camera_create(asset_store=asset_store), scene, asset_store=asset_store)
    image = camera.fields["ip"].data["srgb"]
    output_path = Path("reports/parity/end_to_end.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, image)
    print(output_path)


if __name__ == "__main__":
    main()

