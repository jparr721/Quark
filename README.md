# quark
⚗️Very fast linear algebra tooling, built for humans.

## Overview
This library aims to be a *dead* simple interface for doing linear algebra in rust. Other libraries that I've seen have an unintuitive interface. I want something as simple as numpy, but in rust.

## Architecture
This library aims to remove all of the unnecessary parts of developing and working with linear systems. This is a floating-point only and numerically stable library. There are plans to include bindings to nodejs, but those are pending.

### API
It is trivial to make a 2d matrix and perform operations on it:

```rust
fn main() {
  let mat = Matrix::new(vec![vec![1,2,3], vec![4,5,6]])
  
  // Now, we can perform operations
  // Transpose:
  let transposed_mat = t(mat);
  
  // Gauss-Jordan row reduction
  let rhs = Matrix::new(vec![vec![0.0;3], vec![0.0;3]])
  let reduced_mat = gaussj(mat, rhs);
}
```

More documentation will be added as new functionality is included.
