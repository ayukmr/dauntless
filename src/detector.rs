use crate::config::Config;
use crate::types::Dim;
use crate::ws::{CannyWorkspace, HarrisWorkspace};

#[derive(Default)]
pub struct Detector {
    pub cws: CannyWorkspace,
    pub hws: HarrisWorkspace,
    pub config: Config,
}

impl Detector {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            cws: CannyWorkspace::default(),
            hws: HarrisWorkspace::default(),
        }
    }

    pub fn ensure(&mut self, dim: Dim) {
        self.cws.ensure(dim);
        self.hws.ensure(dim);
    }
}
