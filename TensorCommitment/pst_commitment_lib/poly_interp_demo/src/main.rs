use ark_bn254::Fr;
use ark_ff::{Field, One, PrimeField, Zero};
use ndarray::{ArrayD, IxDyn};
use ndarray_npy::read_npy;
use serde::Serialize;
use std::error::Error;
use std::fs;
use std::time::Instant;
/// cargo run --release -- rand10by10by10by10by10by10.npy
type F = Fr;

macro_rules! log_println {
    ($quiet:expr, $($arg:tt)*) => {{
        if $quiet {
            eprintln!($($arg)*);
        } else {
            println!($($arg)*);
        }
    }};
}

macro_rules! log_print {
    ($quiet:expr, $($arg:tt)*) => {{
        if $quiet {
            eprint!($($arg)*);
        } else {
            print!($($arg)*);
        }
    }};
}

#[derive(Serialize)]
struct CoeffDump {
    dims: Vec<usize>,
    degree_bound: usize,
    coefficients: Vec<String>,
}

fn fr_list_to_strings(coeffs: &[F]) -> Vec<String> {
    coeffs.iter().map(|c| c.into_bigint().to_string()).collect()
}

/// Invert Vandermonde V[i,j] = (i)^j for i,j = 0..n-1 over Fr.
fn invert_1d_vandermonde(n: usize) -> Vec<Vec<F>> {
    let mut aug = vec![vec![F::zero(); 2 * n]; n];

    // Build [V | I]
    for i in 0..n {
        let x = F::from(i as u64);
        let mut pow = F::one();
        for j in 0..n {
            aug[i][j] = pow;
            pow *= x;
        }
        aug[i][n + i] = F::one();
    }

    // Gaussian elimination to get [I | V^{-1}]
    for col in 0..n {
        // Find pivot
        let mut pivot = col;
        while pivot < n && aug[pivot][col].is_zero() {
            pivot += 1;
        }
        if pivot == n {
            panic!("Vandermonde is singular (shouldn't happen for distinct nodes)");
        }
        if pivot != col {
            aug.swap(col, pivot);
        }

        // Normalize pivot row
        let inv_p = aug[col][col].inverse().unwrap();
        for j in col..2 * n {
            aug[col][j] *= inv_p;
        }

        // Copy pivot row to avoid aliasing issues
        let pivot_row = aug[col].clone();

        // Eliminate this column from other rows
        for i in 0..n {
            if i == col {
                continue;
            }
            let factor = aug[i][col];
            if factor.is_zero() {
                continue;
            }
            for j in col..2 * n {
                aug[i][j] -= factor * pivot_row[j];
            }
        }
    }

    // Extract inverse: right half
    let mut inv = vec![vec![F::zero(); n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }
    inv
}

/// Compute row-major strides for dims: strides[i] = product_{j>i} dims[j].
fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let d = dims.len();
    let mut strides = vec![0usize; d];
    if d == 0 {
        return strides;
    }
    strides[d - 1] = 1;
    for i in (0..d - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

/// Convert multi-index -> flat index using precomputed strides.
fn multi_index_to_flat(idx: &[usize], strides: &[usize]) -> usize {
    idx.iter().zip(strides.iter()).map(|(i, s)| i * s).sum()
}

/// Increment multi-index in lexicographical order. Returns false on wrap-around.
fn increment_index(idx: &mut [usize], dims: &[usize]) -> bool {
    let d = dims.len();
    for i in (0..d).rev() {
        idx[i] += 1;
        if idx[i] < dims[i] {
            return true;
        }
        idx[i] = 0;
    }
    false
}

/// Apply 1D transform `inv` along a given axis of an N-D array stored as a flat Vec<F>.
/// Shape is `dims`, strides are row-major `strides`.
fn apply_axis_transform(
    data: &mut [F],
    dims: &[usize],
    strides: &[usize],
    axis: usize,
    inv: &[Vec<F>],
) {
    let d = dims.len();
    let n_axis = dims[axis];
    assert_eq!(inv.len(), n_axis);
    assert_eq!(inv[0].len(), n_axis);

    // Outer dimensions: treat axis as length 1 so we iterate over all combinations
    // of other dims, and for each one we process the line along `axis`.
    let mut dims_outer = dims.to_vec();
    dims_outer[axis] = 1;

    let mut idx_outer = vec![0usize; d];

    loop {
        let flat_base = multi_index_to_flat(&idx_outer, strides);

        // Gather lane along `axis`
        let mut lane = vec![F::zero(); n_axis];
        let step = strides[axis];
        for j in 0..n_axis {
            lane[j] = data[flat_base + j * step];
        }

        // Transform: new_lane[a] = sum_j inv[a][j] * lane[j]
        let mut new_lane = vec![F::zero(); n_axis];
        for a in 0..n_axis {
            let mut s = F::zero();
            for j in 0..n_axis {
                s += inv[a][j] * lane[j];
            }
            new_lane[a] = s;
        }

        // Write back
        for a in 0..n_axis {
            data[flat_base + a * step] = new_lane[a];
        }

        if !increment_index(&mut idx_outer, &dims_outer) {
            break;
        }
    }
}

/// Evaluate the N-D polynomial with coefficients `coeffs` at a grid point `point`.
/// - `dims` = [n0, n1, ..., n_{d-1}]
/// - `coeffs` are in lex order over the same index space.
/// - `point[k]` is the coordinate along axis k (0 .. n_k-1).
fn eval_poly_at(coeffs: &[F], dims: &[usize], strides: &[usize], point: &[usize]) -> F {
    let d = dims.len();
    assert_eq!(point.len(), d);

    // powers[k][e] = (point[k])^e for e=0..dims[k]-1
    let mut pow_tables: Vec<Vec<F>> = Vec::with_capacity(d);
    for (k, &n_k) in dims.iter().enumerate() {
        let base = F::from(point[k] as u64);
        let mut row = vec![F::one(); n_k];
        for e in 1..n_k {
            row[e] = row[e - 1] * base;
        }
        pow_tables.push(row);
    }

    let mut idx = vec![0usize; d];
    let mut val = F::zero();

    loop {
        let flat = multi_index_to_flat(&idx, strides);
        let c = coeffs[flat];
        if !c.is_zero() {
            let mut mon = F::one();
            for k in 0..d {
                mon *= pow_tables[k][idx[k]];
            }
            val += c * mon;
        }

        if !increment_index(&mut idx, dims) {
            break;
        }
    }

    val
}

/// Generic N-D interpolation on BN254 over the grid {0..n0-1}×...×{0..nd-1}.
fn interp_nd(arr: &ArrayD<i64>, quiet: bool) -> (Vec<usize>, Vec<F>) {
    let dims: Vec<usize> = arr.shape().iter().map(|&n| n as usize).collect();
    let d = dims.len();
    if d == 0 {
        panic!("Scalar tensor not supported");
    }

    log_println!(quiet, "==== N-D interpolation over Fr ====");
    log_println!(quiet, "Dims: {:?}", dims);

    let total_points: usize = dims.iter().product();
    log_println!(quiet, "Total grid points: {}", total_points);

    let t0 = Instant::now();

    // Flatten and map to Fr (handles both positive and negative ints)
    let values: Vec<F> = arr
        .iter()
        .map(|&v| {
            if v >= 0 {
                F::from(v as u64)
            } else {
                -F::from((-v) as u64)
            }
        })
        .collect();

    let strides = compute_strides(&dims);

    // Precompute Vandermonde inverses per axis
    let mut inv_mats: Vec<Vec<Vec<F>>> = Vec::with_capacity(d);
    for &n in &dims {
        inv_mats.push(invert_1d_vandermonde(n));
    }

    let t1 = Instant::now();

    // Tensor-product interpolation: apply inverse Vandermonde along each axis
    let mut coeffs = values.clone();
    for axis in 0..d {
        apply_axis_transform(&mut coeffs, &dims, &strides, axis, &inv_mats[axis]);
    }

    let t2 = Instant::now();

    // Print some coefficients in lex order (same as flattened order)
    log_println!(quiet, "\nSome coefficients c_(a_0, ..., a_{}):", d - 1);
    let max_to_print = 20;
    let mut printed = 0;
    let mut idx = vec![0usize; d];

    loop {
        let flat = multi_index_to_flat(&idx, &strides);
        let c = coeffs[flat];
        if !c.is_zero() {
            log_print!(quiet, "c_(");
            for k in 0..d {
                if k > 0 {
                    log_print!(quiet, ", ");
                }
                log_print!(quiet, "{:2}", idx[k]);
            }
            log_println!(quiet, ") = {:?}", c);
            printed += 1;
            if printed >= max_to_print {
                log_println!(quiet, "... (truncated)");
                break;
            }
        }

        if !increment_index(&mut idx, &dims) {
            break;
        }
    }

    // Light sanity checks at a couple of points (not full grid, to keep it fast)
    let mut points: Vec<Vec<usize>> = Vec::new();
    // All zeros
    points.push(vec![0; d]);
    // Center-ish point if dims > 1
    if dims.iter().all(|&n| n > 1) {
        let center: Vec<usize> = dims.iter().map(|&n| n / 2).collect();
        points.push(center);
    }

    let t_check_start = Instant::now();
    log_println!(quiet, "\nSanity checks at a few grid points:");

    for p in &points {
        let true_int = arr[IxDyn(&p[..])];
        let true_fr = if true_int >= 0 {
            F::from(true_int as u64)
        } else {
            -F::from((-true_int) as u64)
        };
        let rec = eval_poly_at(&coeffs, &dims, &strides, p);
        log_println!(quiet, "  at {:?}: true = {:?}, rec = {:?}", p, true_fr, rec);
    }

    let t3 = Instant::now();

    log_println!(quiet, "\n=== Timing (Fr, N-D tensor interpolation) ===");
    log_println!(quiet, "Setup (map + Vandermondes): {:?}", t1 - t0);
    log_println!(quiet, "N-D transforms:              {:?}", t2 - t1);
    log_println!(
        quiet,
        "Sample checks:               {:?}",
        t3 - t_check_start
    );
    log_println!(quiet, "Total:                       {:?}", t3 - t0);

    (dims, coeffs)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <data.npy> [--coeffs <coeffs.json>]", args[0]);
        std::process::exit(1);
    }
    let path = &args[1];

    let mut coeffs_out: Option<String> = None;
    let mut idx = 2;
    while idx < args.len() {
        match args[idx].as_str() {
            "--coeffs" => {
                if idx + 1 >= args.len() {
                    eprintln!("--coeffs requires a file path");
                    std::process::exit(1);
                }
                coeffs_out = Some(args[idx + 1].clone());
                idx += 2;
            }
            flag => {
                eprintln!("Unknown argument: {}", flag);
                std::process::exit(1);
            }
        }
    }

    let quiet = matches!(coeffs_out.as_deref(), Some("-"));

    let arr: ArrayD<i64> = read_npy(path)?;
    let shape = arr.shape().to_vec();

    log_println!(quiet, "Loaded {:?} with shape {:?}", path, shape);

    let (dims, coeffs) = interp_nd(&arr, quiet);
    if let Some(out_path) = coeffs_out {
        let degree_bound = dims.iter().copied().max().unwrap_or(0);
        let dump = CoeffDump {
            dims,
            degree_bound,
            coefficients: fr_list_to_strings(&coeffs),
        };
        let json = serde_json::to_string_pretty(&dump)?;
        if out_path == "-" {
            println!("{}", json);
        } else {
            fs::write(&out_path, json)?;
            log_println!(
                quiet,
                "Saved {} coefficients to {}",
                dump.coefficients.len(),
                out_path
            );
        }
    }

    Ok(())
}
