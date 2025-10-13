mod types;
mod fft;
mod oper;
mod post;
mod mask;
mod tags;
mod decode;
mod process;

pub use types::{Filter, Tag};
use types::Lightness;

pub fn tags(data: Lightness) -> Vec<Tag> {
    process::process(data, Filter::default())
}

pub fn tags_custom(data: Lightness, filter: Filter) -> Vec<Tag> {
    process::process(data, filter)
}
