"""
leakage_guard.py – Partition integrity checks for PATHSNAP model training.

Ensures that training, student-testing and held-out evaluator trips
never overlap. All logic is based on filesystem discovery — no hardcoded
assumption about which trips exist beyond the evaluator-test exclusion list.
"""

import os
import glob
import logging

logger = logging.getLogger(__name__)

# Evaluator-testing trip IDs — NEVER use these for training or evaluation.
EVALUATOR_TEST_TRIP_IDS: set[int] = {
    15, 17, 23, 30, 49, 59, 67, 73, 87, 94,
}


def _discover_trips(raw_partition_dir: str) -> list[int]:
    """Return sorted list of integer trip IDs in a raw data partition directory."""
    dirs = [
        d for d in os.listdir(raw_partition_dir)
        if os.path.isdir(os.path.join(raw_partition_dir, d))
    ]
    trip_ids = []
    for d in dirs:
        try:
            trip_ids.append(int(d))
        except ValueError:
            pass
    return sorted(trip_ids)


def get_train_trips(raw_dir: str = "data/raw") -> list[int]:
    """Return sorted list of training trip IDs discovered from filesystem."""
    path = os.path.join(raw_dir, "train")
    trips = _discover_trips(path)
    # Safety: exclude any evaluator trips that may have slipped in
    trips = [t for t in trips if t not in EVALUATOR_TEST_TRIP_IDS]
    return trips


def get_student_test_trips(raw_dir: str = "data/raw") -> list[int]:
    """Return sorted list of student-test trip IDs discovered from filesystem."""
    path = os.path.join(raw_dir, "student_test")
    trips = _discover_trips(path)
    trips = [t for t in trips if t not in EVALUATOR_TEST_TRIP_IDS]
    return trips


def verify_no_leakage(
    train_trips: list[int],
    val_trips: list[int] | None = None,
    test_trips: list[int] | None = None,
) -> None:
    """
    Assert that all provided trip-ID sets are disjoint and that
    no evaluator-test trips appear in any set.

    Raises
    ------
    AssertionError if any leakage is detected.
    """
    sets = {"train": set(train_trips)}
    if val_trips:
        sets["val"] = set(val_trips)
    if test_trips:
        sets["test"] = set(test_trips)

    names = list(sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = sets[names[i]] & sets[names[j]]
            assert not overlap, (
                f"DATA LEAKAGE: trip IDs in both {names[i]} and {names[j]}: {overlap}"
            )

    for name, s in sets.items():
        eval_overlap = s & EVALUATOR_TEST_TRIP_IDS
        assert not eval_overlap, (
            f"DATA LEAKAGE: evaluator-test trips found in {name}: {eval_overlap}"
        )

    logger.info("Leakage guard: all partitions are disjoint. No evaluator trips exposed.")


def log_partition_summary(
    train_trips: list[int],
    student_test_trips: list[int],
) -> None:
    """Log a human-readable summary of partition sizes."""
    logger.info(
        f"Partition summary: "
        f"train={len(train_trips)} trips, "
        f"student_test={len(student_test_trips)} trips, "
        f"evaluator_test={len(EVALUATOR_TEST_TRIP_IDS)} trips (held out, never loaded)"
    )
    verify_no_leakage(train_trips, student_test_trips)
