use ark_bn254::Fr;
use ark_ff::{Field, One, Zero};
use rand::Rng;
use rayon::prelude::*;
use std::time::Instant;

type F = Fr;

const N: usize = 100; // grid: 0..9 in each dimension

/// Invert the Vandermonde matrix V[i,j] = (i)^j over Fr for i,j = 0..N-1.
/// Returns M = V^{-1} as Vec<Vec<F>>.
fn invert_1d_vandermonde() -> Vec<Vec<F>> {
    // Augmented matrix [V | I] of size N x 2N
    let mut aug = vec![vec![F::zero(); 2 * N]; N];

    for i in 0..N {
        let x = F::from(i as u64);
        let mut pow = F::one();
        for j in 0..N {
            aug[i][j] = pow; // V[i,j] = x^j
            pow *= x;
        }
        // Identity on the right
        aug[i][N + i] = F::one();
    }

    // Gaussian elimination to get [I | V^{-1}]
    for col in 0..N {
        // Find pivot
        let mut pivot = col;
        while pivot < N && aug[pivot][col].is_zero() {
            pivot += 1;
        }
        if pivot == N {
            panic!("Vandermonde is singular, which should not happen for distinct nodes");
        }
        if pivot != col {
            aug.swap(col, pivot);
        }

        // Normalize pivot row
        let inv_p = aug[col][col].inverse().unwrap();
        for j in col..2 * N {
            aug[col][j] *= inv_p;
        }

        // Eliminate column from other rows
        for i in 0..N {
            if i == col {
                continue;
            }
            let factor = aug[i][col];
            if factor.is_zero() {
                continue;
            }
            for j in col..2 * N {
                let tmp = factor * aug[col][j];
                aug[i][j] -= tmp;
            }
        }
    }

    // Extract inverse: M[i][j] = aug[i][N + j]
    let mut inv = vec![vec![F::zero(); N]; N];
    for i in 0..N {
        for j in 0..N {
            inv[i][j] = aug[i][N + j];
        }
    }
    inv
}

fn main() {
    let t_start = Instant::now();

    // 1) Precompute 1D Vandermonde inverse for nodes x=0..9
    let t0 = Instant::now();
    let m_inv = invert_1d_vandermonde();
    let t1 = Instant::now();

    // 2) Generate random integer data 10x10x10 and map to Fr
    let mut rng = rand::thread_rng();

    // store both the original integers and their field versions
    let mut values = vec![vec![vec![F::zero(); N]; N]; N];    // [x][y][z] in Fr
    let mut values_int = vec![vec![vec![0u64; N]; N]; N];     // [x][y][z] as u64

    for x in 0..N {
        for y in 0..N {
            for z in 0..N {
                let r: u64 = rng.gen_range(0..1000); // random int
                values_int[x][y][z] = r;
                values[x][y][z] = F::from(r);
            }
        }
    }
    let t2 = Instant::now();

    // 3) Transform along x: for each fixed (y,z), solve 1D interpolation in x
    // coeff_x[ax][y][z] = coefficient for x^ax at that (y,z)
    let mut coeff_x = vec![vec![vec![F::zero(); N]; N]; N]; // [ax][y][z]

    // Parallelize over ax: each thread owns coeff_x[ax][..][..]
    coeff_x
        .par_iter_mut()
        .enumerate()
        .for_each(|(ax, coeff_x_ax)| {
            for y in 0..N {
                for z in 0..N {
                    let mut s = F::zero();
                    for j in 0..N {
                        s += m_inv[ax][j] * values[j][y][z];
                    }
                    coeff_x_ax[y][z] = s;
                }
            }
        });

    let t3 = Instant::now();

    // 4) Transform along y: for each fixed (ax,z), solve 1D interpolation in y
    // coeff_xy[ax][ay][z] = coefficient for x^ax y^ay at that z
    let mut coeff_xy = vec![vec![vec![F::zero(); N]; N]; N]; // [ax][ay][z]

    coeff_xy
        .par_iter_mut()
        .enumerate()
        .for_each(|(ax, coeff_xy_ax)| {
            for ay in 0..N {
                for z in 0..N {
                    let mut s = F::zero();
                    for j in 0..N {
                        // j is y-index
                        s += m_inv[ay][j] * coeff_x[ax][j][z];
                    }
                    coeff_xy_ax[ay][z] = s;
                }
            }
        });

    let t4 = Instant::now();

    // 5) Transform along z: for each fixed (ax,ay), solve 1D interpolation in z
    // coeff_xyz[ax][ay][az] = coefficient for x^ax y^ay z^az
    let mut coeff_xyz = vec![vec![vec![F::zero(); N]; N]; N]; // [ax][ay][az]

    coeff_xyz
        .par_iter_mut()
        .enumerate()
        .for_each(|(ax, coeff_xyz_ax)| {
            for ay in 0..N {
                for az in 0..N {
                    let mut s = F::zero();
                    for j in 0..N {
                        // j is z-index
                        s += m_inv[az][j] * coeff_xy[ax][ay][j];
                    }
                    coeff_xyz_ax[ay][az] = s;
                }
            }
        });

    let t5 = Instant::now();

    println!("=== Timing (Fr, 10x10x10, random data, tensor + Rayon) ===");
    println!("Vandermonde inverse (1D): {:?}", t1 - t0);
    println!("Populate random values:   {:?}", t2 - t1);
    println!("X-transform (parallel):   {:?}", t3 - t2);
    println!("Y-transform (parallel):   {:?}", t4 - t3);
    println!("Z-transform (parallel):   {:?}", t5 - t4);
    println!("Total up to coeffs:       {:?}", t5 - t_start);

    // 6) Print a few non-zero coefficients in lex order (ax, ay, az)
    println!("\nSome non-zero coefficients in lex order (ax, ay, az):");
    let mut printed = 0;
    'outer_ax: for ax in 0..N {
        for ay in 0..N {
            for az in 0..N {
                let c = coeff_xyz[ax][ay][az];
                if c.is_zero() {
                    continue;
                }
                println!("c_({:2}, {:2}, {:2}) = {:?}", ax, ay, az, c);
                printed += 1;
                if printed >= 20 {
                    println!("... (truncated)");
                    break 'outer_ax;
                }
            }
        }
    }

    // 7) Precompute power tables for evaluation (to speed up full-grid check)
    let t6 = Instant::now();

    let mut x_pows_table = vec![vec![F::one(); N]; N]; // [x][deg]
    let mut y_pows_table = vec![vec![F::one(); N]; N];
    let mut z_pows_table = vec![vec![F::one(); N]; N];

    for x in 0..N {
        let xf = F::from(x as u64);
        for e in 1..N {
            x_pows_table[x][e] = x_pows_table[x][e - 1] * xf;
        }
    }
    for y in 0..N {
        let yf = F::from(y as u64);
        for e in 1..N {
            y_pows_table[y][e] = y_pows_table[y][e - 1] * yf;
        }
    }
    for z in 0..N {
        let zf = F::from(z as u64);
        for e in 1..N {
            z_pows_table[z][e] = z_pows_table[z][e - 1] * zf;
        }
    }

    let t7 = Instant::now();

    // 8) Full grid check: compare original values vs reconstructed polynomial
    // Parallelized over x for speed.
    // println!("\nFull grid check (original vs reconstructed):");

    // let mismatches: usize = (0..N)
    //     .into_par_iter()
    //     .map(|x| {
    //         let mut local_mismatch = 0usize;
    //         for y in 0..N {
    //             for z in 0..N {
    //                 let true_val = values[x][y][z];

    //                 let mut rec = F::zero();
    //                 for ax in 0..N {
    //                     for ay in 0..N {
    //                         for az in 0..N {
    //                             let c = coeff_xyz[ax][ay][az];
    //                             if c.is_zero() {
    //                                 continue;
    //                             }
    //                             rec += c
    //                                 * x_pows_table[x][ax]
    //                                 * y_pows_table[y][ay]
    //                                 * z_pows_table[z][az];
    //                         }
    //                     }
    //                 }

    //                 if true_val != rec {
    //                     local_mismatch += 1;
    //                 }
    //             }
    //         }
    //         local_mismatch
    //     })
    //     .sum();

    // let t8 = Instant::now();

    // if mismatches == 0 {
    //     println!("All {} points matched exactly in Fr ✅", N * N * N);
    // } else {
    //     println!("{} mismatches found ❌", mismatches);
    // }

    // println!(
    //     "Power tables:              {:?}\nFull-grid check (parallel): {:?}\nTotal time:                 {:?}",
    //     t7 - t6,
    //     t8 - t7,
    //     t8 - t_start
    // );
}
