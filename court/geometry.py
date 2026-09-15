import cv2
import numpy as np

COURT_WIDTH = 20.0
COURT_LENGTH = 44.0
NET_Y = COURT_LENGTH / 2
KITCHEN_DEPTH = 7.0

COURT_CORNERS = np.array(
    [[0.0, 0.0], [COURT_WIDTH, 0.0], [COURT_WIDTH, COURT_LENGTH], [0.0, COURT_LENGTH]],
    dtype=np.float32,
)


def _as_points(points):
    pts = np.asarray(points, dtype=np.float32)
    single = pts.ndim == 1
    return pts.reshape(-1, 2), single


ROW_EDGES = (0.0, KITCHEN_DEPTH, KITCHEN_DEPTH + (NET_Y - KITCHEN_DEPTH) / 2, NET_Y)


def _depth_from_net(y, half):
    return NET_Y - y if half == "far" else y - NET_Y


def _y_from_depth(depth, half):
    return NET_Y - depth if half == "far" else NET_Y + depth


class CourtMapper:
    def __init__(self, image_corners, cols=4):
        self.image_corners = np.asarray(image_corners, dtype=np.float32).reshape(4, 2)
        self.H = cv2.getPerspectiveTransform(self.image_corners, COURT_CORNERS)
        self.H_inv = np.linalg.inv(self.H)
        self.rows = len(ROW_EDGES) - 1
        self.cols = cols
        self.n_cells = self.rows * self.cols

    def _transform(self, points, H):
        pts, single = _as_points(points)
        out = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        return out[0] if single else out

    def pixel_to_court(self, points):
        return self._transform(points, self.H)

    def court_to_pixel(self, points):
        return self._transform(points, self.H_inv)

    def in_court(self, court_points):
        pts, single = _as_points(court_points)
        mask = (
            (pts[:, 0] >= 0) & (pts[:, 0] <= COURT_WIDTH)
            & (pts[:, 1] >= 0) & (pts[:, 1] <= COURT_LENGTH)
        )
        return bool(mask[0]) if single else mask

    def side(self, court_point):
        return "far" if court_point[1] < NET_Y else "near"

    def in_kitchen(self, court_point):
        return abs(court_point[1] - NET_Y) <= KITCHEN_DEPTH

    def grid_cell(self, court_point, half=None):
        x, y = float(court_point[0]), float(court_point[1])
        half = half or self.side(court_point)
        depth = _depth_from_net(y, half)
        if not (0 <= x <= COURT_WIDTH and 0 <= depth <= NET_Y):
            return -1
        col = min(int(x / COURT_WIDTH * self.cols), self.cols - 1)
        row = min(int(np.searchsorted(ROW_EDGES, depth, side="right")) - 1, self.rows - 1)
        return row * self.cols + col

    def pixel_to_cell(self, pixel_point, half=None):
        return self.grid_cell(self.pixel_to_court(pixel_point), half)

    def _cell_bounds(self, cell):
        row, col = divmod(cell, self.cols)
        x0, x1 = col * COURT_WIDTH / self.cols, (col + 1) * COURT_WIDTH / self.cols
        return x0, x1, ROW_EDGES[row], ROW_EDGES[row + 1]

    def cell_center(self, cell, half):
        x0, x1, d0, d1 = self._cell_bounds(cell)
        return np.array([(x0 + x1) / 2, _y_from_depth((d0 + d1) / 2, half)], dtype=np.float32)

    def cell_polygon(self, cell, half):
        x0, x1, d0, d1 = self._cell_bounds(cell)
        y0, y1 = _y_from_depth(d0, half), _y_from_depth(d1, half)
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        return self.court_to_pixel(corners)
