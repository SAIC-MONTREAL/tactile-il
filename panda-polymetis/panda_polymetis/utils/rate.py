import time


class Rate:
    def __init__(self, frequency: float) -> None:
        """
        Maintains a constant control rate for the POMDP loop. Replacement for rate object in ROS.

        :param frequency: Polling frequency, in Hz.
        """
        self.period, self.last = 1.0 / frequency, time.time()

    def sleep(self) -> None:
        current_delta = time.time() - self.last
        sleep_time = max(0.0, self.period - current_delta)
        if sleep_time:
            time.sleep(sleep_time)
        self.last = time.time()