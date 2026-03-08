from __future__ import annotations

from pyisetcam import IEPTable, camera_create, iePTable, ie_p_table, oi_create, scene_create


def test_ie_p_table_scene_window_format(asset_store) -> None:
    scene = scene_create("uniform ee", 8, asset_store=asset_store)

    table, window = ie_p_table(scene, format="window")

    assert isinstance(table, IEPTable)
    assert window is None
    assert table.columns == ("Property", "Value", "Units")
    labels = [row[0] for row in table.data]
    assert "Name" in labels
    assert "Mean luminance" in labels
    assert table.title.endswith("for a Scene")


def test_ie_p_table_oi_embed_includes_optics_rows() -> None:
    oi = oi_create()

    table, window = iePTable(oi, format="embed")

    assert window is None
    assert table.columns == ("Property", "Value")
    labels = [row[0] for row in table.data]
    assert "Optics model" in labels
    assert "Compute method" in labels
    assert table.title.endswith("for an Optical Image")


def test_ie_p_table_camera_combines_pipeline_sections(asset_store) -> None:
    camera = camera_create(asset_store=asset_store)

    table, _ = ie_p_table(camera, format="embed")

    labels = [row[0] for row in table.data]
    assert "Display name" in labels
    assert "Size (mm)" in labels
    assert "Compute method" in labels
    assert table.title.endswith("for a Camera")
