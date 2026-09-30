"""Image-to-ground projection.

The only bridge between image-space measurements and physical ones. Without a
calibration nothing in this module runs, and every physical metric is reported
unavailable rather than estimated.

Why a scalar scale is not enough
--------------------------------
It is tempting to calibrate with a single "metres per pixel". That is only valid
for a near-orthographic view. A traffic camera looks along the road at an angle,
so a pixel near the top of the frame spans many times more ground than a pixel
at the bottom. A scalar therefore produces speeds that are wrong by a factor
that varies across the image -- and wrong in a direction that flatters distant
vehicles, which are exactly the ones an operator is least able to sanity-check.

So the supported calibration is a **homography**: a 3x3 matrix mapping image
points onto the road plane. It is exact for any planar surface viewed by a
pinhole camera, which a road under a fixed camera is a good approximation of.

Scalar scale is accepted only when the calibration explicitly declares a
near-orthographic view, and even then it carries reduced confidence.
"""

from __future__ import annotations

import math

from app.domain.observation import CameraCalibration

#: Confidence attached to physical measurements derived through a homography.
#: Below 1.0 because the road is not perfectly planar and the calibration itself
#: carries residual error.
HOMOGRAPHY_CONFIDENCE = 0.85

#: Confidence for the scalar-scale path. Materially lower: it assumes a view
#: geometry that traffic cameras rarely have.
SCALAR_SCALE_CONFIDENCE = 0.45


class GroundProjector:
    """Projects image points onto road-plane metres.

    Construct via :meth:`from_calibration`, which returns ``None`` when no usable
    calibration exists -- making "no projector" the explicit, checkable state
    rather than a silently identity transform.
    """

    def __init__(
        self,
        *,
        homography: tuple[float, ...] | None = None,
        metres_per_pixel: float | None = None,
    ) -> None:
        if homography is None and metres_per_pixel is None:
            msg = "A projector needs either a homography or a metric scale."
            raise ValueError(msg)
        if homography is not None and len(homography) != 9:
            msg = "A homography must be 9 values (row-major 3x3)."
            raise ValueError(msg)
        self._h = homography
        self._scale = metres_per_pixel

    @classmethod
    def from_calibration(cls, calibration: CameraCalibration | None) -> GroundProjector | None:
        """Build a projector, or ``None`` when calibration is absent or unusable."""
        if calibration is None or not calibration.is_calibrated:
            return None
        if calibration.homography is not None:
            return cls(homography=tuple(calibration.homography))
        if calibration.ground_scale_m_per_px is not None:
            return cls(metres_per_pixel=calibration.ground_scale_m_per_px)
        return None

    @property
    def method(self) -> str:
        """How this projector converts, for provenance."""
        return "homography" if self._h is not None else "uniform_scale"

    @property
    def confidence(self) -> float:
        """Confidence in physical values produced through this projector."""
        return HOMOGRAPHY_CONFIDENCE if self._h is not None else SCALAR_SCALE_CONFIDENCE

    def project(self, point: tuple[float, float]) -> tuple[float, float] | None:
        """Map an image point to road-plane metres.

        Returns ``None`` when the point projects to or beyond the horizon, where
        the homography's third coordinate approaches zero. Those points are
        genuinely at infinite ground distance; returning a huge number instead
        would produce a spectacular and entirely fictitious speed.
        """
        u, v = point
        if self._h is None:
            scale = self._scale or 0.0
            return (u * scale, v * scale)

        h = self._h
        x = h[0] * u + h[1] * v + h[2]
        y = h[3] * u + h[4] * v + h[5]
        w = h[6] * u + h[7] * v + h[8]
        if math.isclose(w, 0.0, abs_tol=1e-9):
            return None
        return (x / w, y / w)

    def ground_distance(
        self, start: tuple[float, float], end: tuple[float, float]
    ) -> float | None:
        """Distance in metres between two image points on the road plane."""
        a = self.project(start)
        b = self.project(end)
        if a is None or b is None:
            return None
        return math.hypot(b[0] - a[0], b[1] - a[1])


def mps_to_kph(value: float) -> float:
    """Convert metres per second to kilometres per hour.

    Exists as a named function rather than a literal because mixing m/s and km/h
    is the classic way to get an ETA wrong by 3.6x, and a plausible-looking one
    at that.
    """
    return value * 3.6
