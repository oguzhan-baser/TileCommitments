use ark_ec::{
    pairing::{Pairing, PairingOutput},
    CurveGroup,
};
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use ark_std::cfg_iter;
use std::{fmt::Display, marker::PhantomData};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub trait InnerProduct {
    type LHS: Copy + CanonicalSerialize + Send;
    type RHS: Copy + CanonicalSerialize + Send;
    type Output: Copy + CanonicalSerialize + PartialEq + Eq + Display;

    fn inner_product(left: &[Self::LHS], right: &[Self::RHS]) -> Self::Output;
}

pub struct PointPointInnerProduct<P: Pairing> {
    _pair: PhantomData<P>,
}

impl<P: Pairing> InnerProduct for PointPointInnerProduct<P> {
    type LHS = P::G1;
    type RHS = P::G2;
    type Output = PairingOutput<P>;

    fn inner_product(left: &[Self::LHS], right: &[Self::RHS]) -> Self::Output {
        assert_eq!(left.len(), right.len());
        cfg_multi_pairing::<P>(
            &P::G1::normalize_batch(left),
            &P::G2::normalize_batch(right),
        )
    }
}

/// Equivalent to `P::multi_pairing`, but with more parallelism (if enabled)
pub fn cfg_multi_pairing<P: Pairing>(
    aff_left: &[P::G1Affine],
    aff_right: &[P::G2Affine],
) -> PairingOutput<P> {
    P::multi_pairing(aff_left, aff_right)
    // let left = cfg_iter!(aff_left)
    //     .map(P::G1Prepared::from)
    //     .collect::<Vec<_>>();
    // let right = cfg_iter!(aff_right)
    //     .map(P::G2Prepared::from)
    //     .collect::<Vec<_>>();

    // // We want to process N chunks in parallel where N is the number of threads available
    // #[cfg(feature = "parallel")]
    // let num_chunks = rayon::current_num_threads();
    // #[cfg(not(feature = "parallel"))]
    // let num_chunks = 1;

    // let chunk_size = if num_chunks <= left.len() {
    //     left.len() / num_chunks
    // } else {
    //     // More threads than elements. Just do it all in parallel
    //     1
    // };

    // #[cfg(feature = "parallel")]
    // let (left_chunks, right_chunks) = (left.par_chunks(chunk_size), right.par_chunks(chunk_size));
    // #[cfg(not(feature = "parallel"))]
    // let (left_chunks, right_chunks) = (left.chunks(chunk_size), right.chunks(chunk_size));

    // // Compute all the (partial) pairings and take the product. We have to take the product over
    // // P::TargetField because MillerLoopOutput doesn't impl Product
    // let ml_result = left_chunks
    //     .zip(right_chunks)
    //     .map(|(aa, bb)| P::multi_miller_loop(aa.iter().cloned(), bb.iter().cloned()).0)
    //     .product();

    // P::final_exponentiation(MillerLoopOutput(ml_result))
}

pub struct PointScalarInnerProduct<G: CurveGroup> {
    _projective: PhantomData<G>,
}

impl<G: CurveGroup> InnerProduct for PointScalarInnerProduct<G> {
    type LHS = G;
    type RHS = G::ScalarField;
    type Output = G;

    fn inner_product(left: &[Self::LHS], right: &[Self::RHS]) -> Self::Output {
        assert_eq!(left.len(), right.len());
        // Can unwrap because we did the length check above
        G::msm_unchecked(&G::normalize_batch(left), right)
    }
}

pub struct ScalarPointInnerProduct<G: CurveGroup> {
    _projective: PhantomData<G>,
}

impl<G: CurveGroup> InnerProduct for ScalarPointInnerProduct<G> {
    type LHS = G::ScalarField;
    type RHS = G;
    type Output = G;

    fn inner_product(left: &[Self::LHS], right: &[Self::RHS]) -> Self::Output {
        assert_eq!(left.len(), right.len());
        // Can unwrap because we did the length check above
        G::msm_unchecked(&G::normalize_batch(right), left)
    }
}

pub struct ScalarScalarInnerProduct<F: PrimeField> {
    _field: PhantomData<F>,
}

impl<F: PrimeField> InnerProduct for ScalarScalarInnerProduct<F> {
    type LHS = F;
    type RHS = F;
    type Output = F;

    fn inner_product(left: &[Self::LHS], right: &[Self::RHS]) -> Self::Output {
        assert_eq!(left.len(), right.len());
        cfg_iter!(left).zip(right).map(|(x, y)| *x * y).sum()
    }
}
