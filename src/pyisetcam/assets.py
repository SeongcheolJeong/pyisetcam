"""Upstream snapshot fetching and asset loading."""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat

from .exceptions import MissingAssetError
from .utils import interp_spectra

DEFAULT_UPSTREAM_SHA = "412b9f9bdb3262f2552b96f0e769b5ad6cdff821"
DEFAULT_UPSTREAM_TARBALL_SHA256 = "12a7b97f02e83b2986c1b255d483a8f49fe32063bf5326343cc8a43ec27763a3"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_cache_root() -> Path:
    return Path(os.environ.get("PYISETCAM_CACHE_ROOT", _repo_root() / ".cache"))


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "pyisetcam/0.1.0"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_upstream_snapshot(
    *,
    sha: str = DEFAULT_UPSTREAM_SHA,
    cache_root: Path | str | None = None,
    expected_sha256: str = DEFAULT_UPSTREAM_TARBALL_SHA256,
    force: bool = False,
) -> Path:
    """Download and extract the pinned upstream ISETCam snapshot."""

    cache_root_path = Path(cache_root) if cache_root is not None else _default_cache_root()
    snapshot_dir = cache_root_path / "upstream" / "isetcam" / sha
    archive_path = cache_root_path / "upstream" / "downloads" / f"isetcam-{sha}.tar.gz"
    if snapshot_dir.exists() and not force:
        return snapshot_dir

    if force and snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)

    download_url = f"https://api.github.com/repos/ISET/isetcam/tarball/{sha}"
    if force or not archive_path.exists():
        _download(download_url, archive_path)

    observed_sha256 = _sha256(archive_path)
    if expected_sha256 and observed_sha256 != expected_sha256:
        raise MissingAssetError(
            f"Upstream tarball hash mismatch: expected {expected_sha256}, observed {observed_sha256}"
        )

    snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(snapshot_dir.parent)) as temporary_root:
        temp_root_path = Path(temporary_root)
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(temp_root_path)
        extracted_directories = [path for path in temp_root_path.iterdir() if path.is_dir()]
        if len(extracted_directories) != 1:
            raise MissingAssetError("Unexpected upstream archive layout.")
        shutil.move(str(extracted_directories[0]), snapshot_dir)
    return snapshot_dir


class AssetStore:
    """Resolve and load assets from the pinned upstream snapshot."""

    def __init__(
        self,
        snapshot_root: Path | str | None = None,
        *,
        sha: str = DEFAULT_UPSTREAM_SHA,
        expected_sha256: str = DEFAULT_UPSTREAM_TARBALL_SHA256,
    ) -> None:
        self._sha = sha
        self._expected_sha256 = expected_sha256
        if snapshot_root is None:
            override = os.environ.get("PYISETCAM_UPSTREAM_ROOT")
            self.snapshot_root = Path(override) if override else ensure_upstream_snapshot()
        else:
            self.snapshot_root = Path(snapshot_root)

    @classmethod
    def default(cls) -> "AssetStore":
        return cls()

    def ensure(self) -> Path:
        if not self.snapshot_root.exists():
            self.snapshot_root = ensure_upstream_snapshot(
                sha=self._sha,
                expected_sha256=self._expected_sha256,
            )
        return self.snapshot_root

    def resolve(self, relative_path: str | Path) -> Path:
        relative_path = Path(relative_path)
        if relative_path.exists():
            return relative_path
        candidate = self.ensure() / relative_path
        if not candidate.exists():
            raise MissingAssetError(f"Asset not found: {relative_path}")
        return candidate

    def load_mat(self, relative_path: str | Path) -> dict[str, Any]:
        path = self.resolve(relative_path)
        try:
            return loadmat(path, squeeze_me=True, struct_as_record=False)
        except NotImplementedError:
            with h5py.File(path, "r") as handle:
                return {key: self._decode_h5(handle[key]) for key in handle.keys()}

    def _decode_h5(self, node: h5py.Dataset | h5py.Group) -> Any:
        if isinstance(node, h5py.Dataset):
            data = node[()]
            if isinstance(data, bytes):
                return data.decode("utf-8")
            return data
        return {key: self._decode_h5(node[key]) for key in node.keys()}

    def load_reflectances(
        self,
        surface_file: str = "macbethChart.mat",
        wave_nm: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        relative_path = Path("data/surfaces/reflectances") / surface_file
        data = self.load_mat(relative_path)
        wavelengths = np.asarray(data["wavelength"], dtype=float)
        reflectances = np.asarray(data["data"], dtype=float)
        if wave_nm is not None:
            reflectances = interp_spectra(wavelengths, reflectances, np.asarray(wave_nm, dtype=float))
            wavelengths = np.asarray(wave_nm, dtype=float)
        return wavelengths, reflectances

    def load_illuminant(
        self,
        illuminant_name: str = "D65.mat",
        wave_nm: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        name = illuminant_name
        if not name.lower().endswith(".mat"):
            if name.upper() == "D65":
                name = "D65.mat"
            else:
                name = f"{name}.mat"
        relative_path = Path("data/lights") / name
        data = self.load_mat(relative_path)
        wavelengths = np.asarray(data["wavelength"], dtype=float)
        illuminant = np.asarray(data["data"], dtype=float)
        if wave_nm is not None:
            illuminant = interp_spectra(wavelengths, illuminant, np.asarray(wave_nm, dtype=float))
            wavelengths = np.asarray(wave_nm, dtype=float)
        return wavelengths, np.asarray(illuminant, dtype=float).reshape(-1)

    def load_xyz(
        self,
        *,
        wave_nm: np.ndarray | None = None,
        energy: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        name = "data/human/XYZEnergy.mat" if energy else "data/human/XYZ.mat"
        data = self.load_mat(name)
        wavelengths = np.asarray(data["wavelength"], dtype=float)
        xyz = np.asarray(data["data"], dtype=float)
        if wave_nm is not None:
            xyz = interp_spectra(wavelengths, xyz, np.asarray(wave_nm, dtype=float))
            wavelengths = np.asarray(wave_nm, dtype=float)
        return wavelengths, xyz

    def load_xyz_quanta(
        self,
        *,
        wave_nm: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        data = self.load_mat("data/human/xyzQuanta.mat")
        wavelengths = np.asarray(data["wavelength"], dtype=float)
        xyz = np.asarray(data["data"], dtype=float)
        if wave_nm is not None:
            xyz = interp_spectra(wavelengths, xyz, np.asarray(wave_nm, dtype=float))
            wavelengths = np.asarray(wave_nm, dtype=float)
        return wavelengths, xyz

    def load_thibos_virtual_eyes(
        self,
        pupil_diameter_mm: float = 6.0,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        pupil = float(np.asarray(pupil_diameter_mm, dtype=float).reshape(-1)[0])
        candidates = {
            7.5: "data/optics/thibosvirtualeyes/IASstats75.mat",
            6.0: "data/optics/thibosvirtualeyes/IASstats60.mat",
            4.5: "data/optics/thibosvirtualeyes/IASstats45.mat",
            3.0: "data/optics/thibosvirtualeyes/IASstats30.mat",
        }
        matched = next((size for size in candidates if np.isclose(pupil, size)), None)
        if matched is None:
            raise MissingAssetError(
                f"Unsupported Thibos pupil size {pupil:.4g} mm. Options are 3.0, 4.5, 6.0, and 7.5."
            )

        data = self.load_mat(candidates[matched])
        sample_mean = np.asarray(data["sample_mean"], dtype=float).reshape(-1).copy()
        sample_cov = np.asarray(data["S"], dtype=float).copy()
        subject_coeffs = {
            "left_eye": np.asarray(data["OS"], dtype=float).copy(),
            "right_eye": np.asarray(data["OD"], dtype=float).copy(),
            "both_eyes": np.asarray(data["OU"], dtype=float).T.copy(),
        }
        return sample_mean, sample_cov, subject_coeffs

    def load_color_filters(
        self,
        filter_name: str,
        wave_nm: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        normalized = filter_name.lower()
        mapping = {
            "rgb": "data/sensor/colorfilters/RGB.mat",
            "rgbw": "data/sensor/colorfilters/RGBW.mat",
            "interleavedrgbw": "data/sensor/colorfilters/interleavedRGBW.mat",
            "cym": "data/sensor/colorfilters/cym.mat",
            "grbc": "data/sensor/colorfilters/GRBC.mat",
            "r": "data/sensor/colorfilters/R.mat",
            "w": "data/sensor/colorfilters/W.mat",
            "mt9v024rgb": "data/sensor/colorfilters/auto/MT9V024_RGB.mat",
            "mt9v024mono": "data/sensor/colorfilters/auto/MT9V024_Mono.mat",
            "mt9v024rgbw": "data/sensor/colorfilters/auto/MT9V024_RGBW.mat",
            "mt9v024rccc": "data/sensor/colorfilters/auto/MT9V024_RCCC.mat",
            "ar0132atrgb": "data/sensor/colorfilters/auto/ar0132atRGB.mat",
            "ar0132atrgbw": "data/sensor/colorfilters/auto/ar0132atRGBW.mat",
            "ar0132atrccc": "data/sensor/colorfilters/auto/ar0132atRCCC.mat",
            "monochrome": None,
        }
        if normalized == "xyz":
            wavelengths, xyz = self.load_xyz_quanta(wave_nm=wave_nm)
            return wavelengths, xyz / np.max(xyz), ["rX", "gY", "bZ"]
        if normalized == "monochrome":
            if wave_nm is None:
                raise MissingAssetError("A target wave is required for monochrome filters.")
            wave = np.asarray(wave_nm, dtype=float)
            return wave, np.ones((wave.size, 1), dtype=float), ["w"]
        relative_path = mapping.get(normalized, filter_name)
        if relative_path is None:
            raise MissingAssetError(f"Unknown filter set: {filter_name}")
        data = self.load_mat(relative_path)
        wavelengths = np.asarray(data["wavelength"], dtype=float)
        filters = np.asarray(data["data"], dtype=float)
        if filters.ndim == 1:
            filters = filters.reshape(-1, 1)
        names = [str(value) for value in np.atleast_1d(data.get("filterNames", []))]
        if not names:
            names = [f"f{index + 1}" for index in range(filters.shape[1])]
        if wave_nm is not None:
            filters = interp_spectra(wavelengths, filters, np.asarray(wave_nm, dtype=float))
            wavelengths = np.asarray(wave_nm, dtype=float)
        return wavelengths, filters, names

    def load_display_struct(self, display_name: str) -> Any:
        name = display_name
        if not name.lower().endswith(".mat"):
            name = f"{name}.mat"
        data = self.load_mat(Path("data/displays") / name)
        return data["d"]
