from PyExpUtils.utils.permute import Record
from typing import Any, Callable, Dict, List, cast
import numpy as np
from PyExpUtils.utils.arrays import last, fillRest_
class Collector:
    def __init__(self):
        self.run_data: Record = {}
        self.all_data: Record = {}

        self.sample_rate: Dict[str, int] = {}
        self.sample_clock: Dict[str, int] = {}

    def reset(self):
        for k in self.run_data:
            # if there's already an array get that
            # otherwise construct a new empty array
            arr = self.all_data.get(k, [])
            arr.append(self.run_data[k])

            # put the array back in case we were working with a new array
            self.all_data[k] = arr

        # reset the run_data for the next run
        self.run_data = {}

    def fillRest(self, name: str, steps: int):
        arr = self.run_data[name]
        l = last(arr)
        fillRest_(arr, l, steps)

    def collect(self, name: str, value: Any):
        sample_rate = self.sample_rate.get(name, 1)
        sample_clock = self.sample_clock.get(name, 0)
        self.sample_clock[name] = sample_clock + 1

        if sample_clock % sample_rate > 0:
            return

        arr = self.run_data.get(name, [])
        arr.append(value)

        self.run_data[name] = arr

    def concat(self, name: str, values: List[Any]):
        # don't just append to the array
        # we need to make sure we respect sample rates
        for v in values:
            self.collect(name, v)

    def evaluate(self, name: str, lmbda: Callable[[], Any]):
        sample_rate = self.sample_rate.get(name, 1)
        sample_clock = self.sample_clock.get(name, 0)
        self.sample_clock[name] = sample_clock + 1

        if sample_clock % sample_rate > 0:
            return

        value = lmbda()
        arr = self.run_data.get(name, [])
        arr.append(value)

        self.run_data[name] = arr

    def setSampleRate(self, name: str, every: int):
        self.sample_rate[name] = every
        self.sample_clock[name] = 0

    def getStats(self, name: str):
        arr = self.all_data[name]

        runs = len(arr)
        min_len = min(map(lambda a: len(a), arr))

        arr = list(map(lambda a: a[:min_len], arr))
        mean: float = cast(float, np.mean(arr, axis=0))
        stderr: float = cast(float, np.std(arr, axis=0, ddof=1)) / np.sqrt(runs)

        return (mean, stderr, runs)

class FilteredCollector(Collector):
    def __init__(self, save_keys: list):
        """Collector where non-saved keys are filtered

        Args:
            save_keys (list): A list of keys to save. Other keys are filtered
        """

        super().__init__()
        self.save_keys = save_keys
    
    def collect(self, name: str, value: Any):
        if name in self.save_keys:
            super().collect(name, value)
        # Ignore if not set to save

    def has_key(self, key):
        return key in self.save_keys