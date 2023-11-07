# zkSaaS: Zero-Knowledge SNARKs as a Service [ePrint:2023/905](https://eprint.iacr.org/2023/905)

Rust implementation of the zkSaaS protocol introduced in [ePrint:2023/905](https://eprint.iacr.org/2023/905)

**WARNING:** This is an academic proof-of-concept prototype, and in particular has not received careful code review. This implementation is NOT ready for production use.

**Update:** A production ready version of zksaas is currently being built out at [webb-tools/zk-SaaS](https://github.com/webb-tools/zk-SaaS)

## Dependencies
This project relies on the [arkworks](http://arkworks.rs) project for finite field and elliptic curve arithmetic. For communication we use the mpc-net crate from [collaborative-zksnark](https://github.com/alex-ozdemir/collaborative-zksnark).

## Overview
* [`secret-sharing/`](secret-sharing): A packed secret sharing library built on top of the finite field generics in arkworks.
* [`dist-primitives/`](dist-primitives): Contains implementations of the distributed fast-fourier transform, multiscalar multiplication and partial products, complete with correctness tests.
* [`groth16/`](groth16): Contains a distributed and local version of groth16 used for benchmarking timings. (not a complete implementation)
* [`plonk/`](plonk): Contains a distributed and local version of plonk used for benchmarking timings. (not a complete implementation)
* [`scripts/`](scripts): Contains shell scripts to run various tests and benchmarks.

## License
This library is released under the MIT License.
