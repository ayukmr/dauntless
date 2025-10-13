mod types;
mod fft;
mod oper;
mod post;
mod mask;
mod tags;
mod decode;
mod process;

pub use types::Tag;
use types::{Lightness};

pub fn tags(data: Lightness) -> Vec<Tag> {
    process::process(data)
}
