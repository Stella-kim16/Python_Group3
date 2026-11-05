"""Domain models and Abstract Base Classes for Subsystem 1: Event-Driven Stream Processing."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional


class StreamProcessor(ABC):
    """
    Abstract Base Class (ABC) defining the uniform interface contract for all
    event-driven stream processors in the Smart Energy Management System.

    Demonstrates:
    1. Abstract Base Classes (ABCs): @abstractmethod `process`.
    2. Inheritance & Polymorphism: Uniform API across filters and transformers.
    3. Operator Overloading: `__add__` (+) and `__or__` (|) to compose pipelines.
    """

    @abstractmethod
    def process(self, data: Any) -> Any:
        """
        Process a single streaming event (dict / EnergyLog) or batch DataFrame.
        Returns None if the event is filtered out (e.g., inactive or sub-threshold meter).
        """
        pass

    def __add__(self, other: "StreamProcessor") -> "StreamPipeline":
        """Compose two stream processors into a pipeline using the `+` operator."""
        if not isinstance(other, StreamProcessor):
            return NotImplemented
        left_steps = self.processors if isinstance(self, StreamPipeline) else [self]
        right_steps = other.processors if isinstance(other, StreamPipeline) else [other]
        return StreamPipeline(left_steps + right_steps)

    def __or__(self, other: "StreamProcessor") -> "StreamPipeline":
        """Compose two stream processors into a pipeline using the `|` operator."""
        return self.__add__(other)

    def __call__(self, data: Any) -> Any:
        """Allow processor instances to be called directly as functions."""
        return self.process(data)


class StreamPipeline(StreamProcessor):
    """
    Composable event-driven stream processing pipeline created by chaining
    `StreamProcessor` instances with `+` (`__add__`) or `|` (`__or__`).

    Supports both single-event polymorphic execution and lazy generator/iterator
    stream consumption (`__iter__` and `__next__`).
    """

    def __init__(
        self,
        processors: Optional[List[StreamProcessor]] = None,
        source: Optional[Iterable[Any]] = None,
    ) -> None:
        self.processors: List[StreamProcessor] = list(processors or [])
        self._source: Optional[Iterable[Any]] = source
        self._iterator: Optional[Iterator[Any]] = None

    def bind_source(self, source: Iterable[Any]) -> "StreamPipeline":
        """Bind an iterable or stream source to this pipeline for iterator consumption."""
        self._source = source
        self._iterator = None
        return self

    def process(self, data: Any) -> Any:
        """
        Pass an event (or DataFrame) sequentially through all composed processors.
        If any filter stage returns None, short-circuit and drop the event.
        """
        current = data
        for processor in self.processors:
            if current is None:
                return None
            current = processor.process(current)
        return current

    def process_stream(self, stream: Iterable[Any]) -> Iterator[Any]:
        """
        Generator that lazily consumes an input stream on the fly and yields
        only valid, transformed events (skipping filtered-out None events).
        """
        for event in stream:
            result = self.process(event)
            if result is not None:
                yield result

    def __iter__(self) -> "StreamPipeline":
        """Initialize stream iteration over the bound source."""
        if self._source is None:
            raise ValueError("No stream source bound to StreamPipeline. Call `bind_source(stream)` first.")
        self._iterator = iter(self._source)
        return self

    def __next__(self) -> Any:
        """
        Fetch and process the next valid event from the bound stream source,
        automatically skipping events filtered out (where `process` returns None).
        """
        if self._iterator is None:
            raise StopIteration
        while True:
            raw_event = next(self._iterator)  # Raises StopIteration when exhausted
            processed = self.process(raw_event)
            if processed is not None:
                return processed


@dataclass
class Building:
    """Represents a monitored smart building (Home, Office, or University)."""

    building_id: str
    building_type: str
    area_sqft: float
    max_occupancy: int
    baseline_kwh: float

    def get_building_info(self) -> Dict[str, Any]:
        """Return building metadata as a dictionary."""
        return asdict(self)


@dataclass
class EnergyLog:
    """Represents a single validated and feature-engineered energy telemetry record."""

    timestamp: datetime
    building_id: str
    building_type: str
    area_sqft: float
    temperature: float
    humidity: float
    occupancy: int
    hvac_usage: str
    lighting_usage: str
    renewable_energy_pct: float
    day_of_week: str
    holiday: str
    energy_consumption_kwh: float
    carbon_footprint_kg: float = 0.0
    energy_cost_eur: float = 0.0
    is_anomaly: bool = False

    def validate_data_types(self) -> bool:
        """Validate domain constraints and data types for this telemetry record."""
        if not isinstance(self.timestamp, datetime):
            return False
        if not isinstance(self.occupancy, int) or self.occupancy < 0:
            return False
        if self.area_sqft <= 0:
            return False
        if not (0.0 <= self.humidity <= 100.0):
            return False
        if not (0.0 <= self.renewable_energy_pct <= 100.0):
            return False
        if self.hvac_usage not in ("On", "Off") or self.lighting_usage not in ("On", "Off"):
            return False
        if self.holiday not in ("Yes", "No"):
            return False
        if self.energy_consumption_kwh < 0:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the energy log record into a JSON-compatible dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return data
