"""Errors are fatal; callers must not silently discard a failed trajectory."""


class PartitionError(ValueError):
    pass


class SCFConvergenceError(RuntimeError):
    pass
