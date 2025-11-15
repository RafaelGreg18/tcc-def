from collections import deque


class RollingSlope:
    """
    A window of points (x, y) e calc a slope OLS:
      slope = (n*Sxy - Sx*Sy) / (n*Sxx - Sx^2)
    Update in O(1) by round (removing oldest).
    """
    def __init__(self, window: int):
        if window < 2:
            raise ValueError("window must be >= 2.")
        self.window = window
        self.buf = deque()  # store (x, y)
        self.Sx = 0.0
        self.Sy = 0.0
        self.Sxx = 0.0
        self.Sxy = 0.0

    def _add(self, x: float, y: float) -> None:
        self.buf.append((x, y))
        self.Sx += x
        self.Sy += y
        self.Sxx += x * x
        self.Sxy += x * y

    def _remove_oldest(self) -> None:
        x0, y0 = self.buf.popleft()
        self.Sx -= x0
        self.Sy -= y0
        self.Sxx -= x0 * x0
        self.Sxy -= x0 * y0

    def push(self, x: float, y: float) -> None:
        if len(self.buf) == self.window:
            self._remove_oldest()
        self._add(x, y)

    def slope(self) -> float:
        n = len(self.buf)
        if n < 2:
            return 0.0
        denom = n * self.Sxx - self.Sx * self.Sx
        if denom == 0.0:
            return 0.0
        return (n * self.Sxy - self.Sx * self.Sy) / denom
