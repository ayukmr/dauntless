use ndarray::Array2;
use num_complex::Complex;

pub type Lightness = Array2<f32>;
pub type Frequency = Array2<Complex<f32>>;
pub type Mask = Array2<bool>;

pub type Point = (u32, u32);
pub type Points = Vec<Point>;
pub type Shapes = Vec<Points>;
pub type Corners = (Point, Point, Point, Point);
