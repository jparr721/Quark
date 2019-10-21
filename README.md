# quark
⚗️Very fast linear algebra tooling, built for humans.

## Overview
This library aims to be a *dead* simple interface for doing linear algebra in rust. Other libraries that I've seen have an unintuitive interface. I want something as simple as numpy, but in rust. That being said, there are a few tradeoffs, namely, all containers are now dynamic. This is in an effort to reduce the overhead with making a vector that just *works*. The library gives control of making the size static etc to the user. Performance optimizations will be flexibly allowed.
