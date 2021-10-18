"""
Time managing class, includying the management of control variables:
initial_time, final_time, time_step and saveper.
"""


class Time(object):
    def __init__(self):
        self._time = None
        self.stage = None
        self.return_timestamps = None

    def __call__(self):
        return self._time

    def set_control_vars(self, **kwargs):
        """
        Set the control variables valies

        Parameters
        ----------
        **kwards:
            initial_time: float, callable or None
                Initial time.
            final_time: float, callable or None
                Final time.
            time_step: float, callable or None
                Time step.
            saveper: float, callable or None
                Saveper.

        """
        def _convert_value(value):
            # this function is necessary to avoid copying the pointer in the
            # lambda function.
            if callable(value):
                return value
            else:
                return lambda: value

        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, _convert_value(value))

        if "initial_time" in kwargs:
            self._initial_time = self.initial_time()
            self._time = self.initial_time()

    def in_bounds(self):
        """
        Check if time is smaller than current final time value.

        Returns
        -------
        bool:
            True if time is smaller than final time. Otherwise, returns Fase.

        """
        return self._time < self.final_time()

    def in_return(self):
        """ Check if current time should be returned """
        if self.return_timestamps is not None:
            return self._time in self.return_timestamps

        return (self._time - self._initial_time) % self.saveper() == 0

    def add_return_timestamps(self, return_timestamps):
        """ Add return timestamps """
        if return_timestamps is None or hasattr(return_timestamps, '__len__'):
            self.return_timestamps = return_timestamps
        else:
            self.return_timestamps = [return_timestamps]

    def update(self, value):
        """ Update current time value """
        self._time = value

    def reset(self):
        """ Reset time value to the initial """
        self._time = self._initial_time
