use crate::types::Point2D;

pub struct Homography {
    pub mat: [f64; 9],
}

impl Homography {
    pub fn from_corners(
        (
            Point2D(x0, y0),
            Point2D(x1, y1),
            Point2D(x2, y2),
            Point2D(x3, y3),
        ): (Point2D, Point2D, Point2D, Point2D),
    ) -> Homography {
        let dx1 = x1 - x3;
        let dx2 = x2 - x3;
        let dx3 = x0 - x1 + x3 - x2;

        let dy1 = y1 - y3;
        let dy2 = y2 - y3;
        let dy3 = y0 - y1 + y3 - y2;

        let denom = dx1 * dy2 - dx2 * dy1;

        let g = (dx3 * dy2 - dx2 * dy3) / denom;
        let h = (dx1 * dy3 - dx3 * dy1) / denom;

        Homography {
            mat: [
                x1 - x0 + g * x1, x2 - x0 + h * x2, x0,
                y1 - y0 + g * y1, y2 - y0 + h * y2, y0,
                g, h, 1.0,
            ],
        }
    }

    pub fn map(&self, u: f64, v: f64) -> Point2D {
        let m = &self.mat;

        let x = m[0] * u + m[1] * v + m[2];
        let y = m[3] * u + m[4] * v + m[5];
        let w = m[6] * u + m[7] * v + m[8];

        Point2D(x / w, y / w)
    }
}
