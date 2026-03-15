#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Tag {
    pub id: Option<u32>,
    pub rot: f32,
    pub pos: Point3D,
    pub corners: Corners,
}

#[derive(Clone, Copy)]
pub struct Dim {
    pub w: usize,
    pub h: usize,
}

impl Dim {
    pub fn len(&self) -> usize {
        self.w * self.h
    }
}

pub type Lightness = Vec<f32>;
pub type Mask = Vec<u8>;
pub type Bits = Vec<bool>;

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Point2D(pub f64, pub f64);

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Point3D(pub f64, pub f64, pub f64);

impl Point3D {
    pub fn dot(&self, o: Point3D) -> f64 {
        self.0 * o.0 + self.1 * o.1 + self.2 * o.2
    }

    pub fn norm(&self) -> f64 {
        self.dot(*self).sqrt()
    }

    pub fn scale(&self, s: f64) -> Point3D {
        Point3D(self.0 * s, self.1 * s, self.2 * s)
    }

    pub fn sub(&self, o: Point3D) -> Point3D {
        Point3D(self.0 - o.0, self.1 - o.1, self.2 - o.2)
    }

    pub fn cross(&self, o: Point3D) -> Point3D {
        Point3D(self.1 * o.2 - self.2 * o.1, self.2 * o.0 - self.0 * o.2, self.0 * o.1 - self.1 * o.0)
    }

    pub fn normed(&self) -> Point3D {
        self.scale(1.0 / self.norm())
    }
}

pub type Corners = (Point2D, Point2D, Point2D, Point2D);
pub type Quads = Vec<Corners>;

pub type Points = Vec<Point2D>;
pub type Shapes = Vec<Points>;
