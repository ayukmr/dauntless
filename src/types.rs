use ndarray::Array2;
use num_complex::Complex;

#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Tag {
    pub id: Option<u32>,
    pub deg: i8,
    pub pos: Point3D,
    pub corners: Corners,
}

pub struct Filter {
    pub quads: bool,
    pub paras: bool,
    pub enclosed: bool,
}

impl Default for Filter {
    fn default() -> Self {
        Self { quads: true, paras: true, enclosed: true }
    }
}

pub type Lightness = Array2<f32>;
pub type Frequency = Array2<Complex<f32>>;
pub type Mask = Array2<bool>;

pub type Point = (u32, u32);
pub type Corners = (Point, Point, Point, Point);

pub type Points = Vec<Point>;
pub type Shapes = Vec<Points>;

pub type FPoint = (f32, f32);
pub type FCorners = (FPoint, FPoint, FPoint, FPoint);

pub type Point2D = (f32, f32);
pub type Point3D = (f32, f32, f32);
