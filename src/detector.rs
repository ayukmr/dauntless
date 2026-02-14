use crate::Config;

pub struct Detector {
    pub config: Config,
}

impl Detector {
    pub fn new(config: Config) -> Self {
        Self { config }
    }
}
