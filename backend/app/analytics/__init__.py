"""Traffic analytics: perception output to measurable traffic behaviour.

    detections + tracks  ->  TrafficAnalyticsEngine  ->  TrafficAnalytics
                         ->  AnalyticsTrafficStateExtractor -> JunctionObservation

Answers "what measurable traffic behaviour is occurring?" and stops there. No
congestion level, no travel time, no route, no recommendation: those interpret
measurements rather than produce them, and belong to later phases.

Image-space and physical quantities are strictly separated. Anything in metres,
m/s or km/h requires a camera calibration; without one those metrics report as
unavailable with a reason, never as zero.
"""

from app.analytics.engine import (
    AnalyticsContext,
    ObservedFrame,
    ObservedObject,
    TrafficAnalyticsEngine,
)
from app.analytics.extractor import AnalyticsTrafficStateExtractor
from app.analytics.schemas import (
    MetricAvailability,
    MetricValue,
    TrafficAnalytics,
)

__all__ = [
    "AnalyticsContext",
    "AnalyticsTrafficStateExtractor",
    "MetricAvailability",
    "MetricValue",
    "ObservedFrame",
    "ObservedObject",
    "TrafficAnalytics",
    "TrafficAnalyticsEngine",
]
