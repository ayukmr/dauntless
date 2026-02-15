#[derive(Debug, Copy, Clone)]
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
        return self.w * self.h;
    }
}

pub type Lightness = Vec<f32>;
pub type Mask = Vec<u8>;
pub type Bits = Vec<bool>;

pub type Point = (u32, u32);
pub type Corners = (Point, Point, Point, Point);
pub type Quads = Vec<Corners>;

pub type Points = Vec<Point>;
pub type Shapes = Vec<Points>;

pub type FPoint = (f32, f32);
pub type FCorners = (FPoint, FPoint, FPoint, FPoint);

pub type Point2D = (f32, f32);
pub type Point3D = (f32, f32, f32);
