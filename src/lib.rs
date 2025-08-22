mod types;
mod fft;
mod oper;
mod post;
mod mask;
mod tags;
mod decode;
mod process;

use types::{Lightness, Corners};

pub fn tags(data: Lightness) -> Vec<(Option<u8>, Option<i8>, Corners)> {
    process::process(data)
}
