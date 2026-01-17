pub struct UnionFind {
    parent: Vec<u32>,
    height: Vec<u32>,
}

impl UnionFind {
    pub fn new() -> Self {
        Self {
            parent: vec![0],
            height: vec![1]
        }
    }

    pub fn push(&mut self, x: u32) {
        self.parent.push(x);
        self.height.push(1);
    }

    pub fn find(&mut self, x: u32) -> u32 {
        let xi = x as usize;
        let p = self.parent[xi];

        if p != x {
            self.parent[xi] = self.find(p);
        }
        self.parent[xi]
    }

    pub fn unite(&mut self, x: u32, y: u32) {
        let xr = self.find(x);
        let yr = self.find(y);

        if xr == yr {
            return;
        }
        let xi = xr as usize;
        let yi = yr as usize;

        if self.height[yi] > self.height[xi] {
            self.parent[xi] = yr;
            self.height[yi] += self.height[xi];
        } else {
            self.parent[yi] = xr;
            self.height[xi] += self.height[yi];
        }
    }
}
