use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{BigInteger, Field, PrimeField};
use ark_std::{One, UniformRand, Zero};
use num_integer::Integer;
use rand::thread_rng;
use rayon::prelude::*;
use spongefish::codecs::arkworks_algebra::{
    FieldToUnitDeserialize, FieldToUnitSerialize, GroupToUnitDeserialize, GroupToUnitSerialize,
    ProverState, UnitToField, VerifierState,
};

pub struct Proof<C: AffineRepr> {
    j: Vec<C>,
    k: Vec<C>,
    l_x: Vec<C::ScalarField>,
    r_x: Vec<C::ScalarField>,
    rho_x: C::ScalarField,
    tau_x: C::ScalarField,
}

pub fn prove<E: Pairing>(
    prover_state: &mut ProverState,
    g: E::G1,
    gs: &[E::G1Affine],
    h: E::G1,
    hs: &[E::G1Affine],
    v: &[E::ScalarField],
    w: E::ScalarField,
    s: &[usize],
    l: usize,
) -> Proof<E::G1Affine> {
    let rng = &mut thread_rng();
    let mut bits = vec![];
    for &i in s {
        let e = v[i];
        let e = e.into_bigint().to_bits_le();
        debug_assert!(e.len() >= l);
        debug_assert!(e[l..].iter().all(|b| !b));
        bits.extend_from_slice(&e[..l]);
    }
    let j = (0..bits.len())
        .map(|_| E::G1Affine::rand(rng))
        .collect::<Vec<_>>();
    let k = (0..bits.len())
        .map(|_| E::G1Affine::rand(rng))
        .collect::<Vec<_>>();
    let rho_l = (0..bits.len())
        .map(|_| E::ScalarField::rand(rng))
        .collect::<Vec<_>>();
    let rho_r = (0..bits.len())
        .map(|_| E::ScalarField::rand(rng))
        .collect::<Vec<_>>();
    let rho_1 = E::ScalarField::rand(rng);
    let rho_2 = E::ScalarField::rand(rng);
    let a = bits
        .iter()
        .zip(&j)
        .zip(&k)
        .map(|((&bit, &j), k)| {
            if bit {
                j
            } else {
                (-k.into_group()).into_affine()
            }
        })
        .sum::<E::G1>()
        + h * rho_1;
    let b = E::G1::msm_unchecked(&j, &rho_l) + E::G1::msm_unchecked(&k, &rho_r) + h * rho_2;

    prover_state.add_points(&[a, b]).unwrap();

    let [y, z]: [E::ScalarField; 2] = prover_state.challenge_scalars().unwrap();

    let l_0 = bits
        .par_iter()
        .map(|&bit| E::ScalarField::from(bit) - z)
        .collect::<Vec<_>>();
    let l_1 = rho_l;
    let r_0 = bits
        .par_iter()
        .enumerate()
        .map(|(i, &bit)| {
            let (j, k) = i.div_rem(&l);
            y.pow([i as u64]) * (E::ScalarField::from(bit) - E::ScalarField::one() + z)
                + E::ScalarField::from(1u128 << k) * z.pow([j as u64 + 1])
        })
        .collect::<Vec<_>>();
    let r_1 = rho_r
        .par_iter()
        .enumerate()
        .map(|(i, &r)| y.pow([i as u64]) * r)
        .collect::<Vec<_>>();
    let t_1 = l_0
        .par_iter()
        .zip(&r_1)
        .chain(l_1.par_iter().zip(&r_0))
        .map(|(&l, r)| l * r)
        .sum::<E::ScalarField>();
    let t_2 = l_1
        .par_iter()
        .zip(&r_1)
        .map(|(&l, r)| l * r)
        .sum::<E::ScalarField>();
    let tau_1 = E::ScalarField::rand(rng);
    let tau_2 = E::ScalarField::rand(rng);
    let tt_1 = g * t_1 + h * tau_1;
    let tt_2 = g * t_2 + h * tau_2;
    let gamma = E::ScalarField::rand(rng);
    let c = g * s
        .par_iter()
        .enumerate()
        .map(|(j, &i)| z.pow([j as u64 + 1]) * v[i])
        .sum::<E::ScalarField>()
        + h * gamma;
    let n = s.len();
    let cc = s
        .par_iter()
        .enumerate()
        .map(|(k, &i)| {
            (0..n)
                .map(|j| {
                    if j == i {
                        Zero::zero()
                    } else {
                        gs[n - i + j] * (z.pow([k as u64 + 1]) * v[j])
                    }
                })
                .sum::<E::G1>()
                + hs[n - 1 - i] * (z.pow([k as u64 + 1]) * w)
        })
        .sum::<E::G1>()
        - hs[n - 1] * gamma;
    let varrho_1 = E::ScalarField::rand(rng);
    let varrho_2 = E::ScalarField::rand(rng);
    let d = g * varrho_1 + h * varrho_2;

    prover_state.add_points(&[tt_1, tt_2, c, cc, d]).unwrap();

    let [eta]: [E::ScalarField; 1] = prover_state.challenge_scalars().unwrap();

    let b1 = varrho_1
        - eta
            * s.par_iter()
                .enumerate()
                .map(|(j, &i)| z.pow([j as u64 + 1]) * v[i])
                .sum::<E::ScalarField>();
    let b2 = varrho_2 - eta * gamma;

    prover_state.add_scalars(&[b1, b2]).unwrap();

    let [x]: [E::ScalarField; 1] = prover_state.challenge_scalars().unwrap();

    let l_x = l_0
        .par_iter()
        .zip(&l_1)
        .map(|(a, b)| x * b + a)
        .collect::<Vec<_>>();
    let r_x = r_0
        .par_iter()
        .zip(&r_1)
        .map(|(a, b)| x * b + a)
        .collect::<Vec<_>>();
    let rho_x = rho_1 + rho_2 * x;
    let tau_x = gamma + tau_1 * x + tau_2 * x.square();

    Proof {
        j,
        k,
        l_x,
        r_x,
        rho_x,
        tau_x,
    }
}

pub fn verify<E: Pairing>(
    verifier_state: &mut VerifierState,
    g: E::G1,
    h: E::G1,
    gg: E::G2,
    ggs: &[E::G2Affine],
    cm: E::G1,
    s: &[usize],
    l: usize,
    proof: Proof<E::G1Affine>,
) {
    let n = s.len();

    let Proof {
        j,
        k,
        l_x,
        r_x,
        rho_x,
        tau_x,
    } = proof;

    let [a, b]: [E::G1; 2] = verifier_state.next_points().unwrap();

    let [y, z]: [E::ScalarField; 2] = verifier_state.challenge_scalars().unwrap();

    let [tt_1, tt_2, c, cc, d]: [E::G1; 5] = verifier_state.next_points().unwrap();
    let [eta]: [E::ScalarField; 1] = verifier_state.challenge_scalars().unwrap();

    let [b1, b2]: [E::ScalarField; 2] = verifier_state.next_scalars().unwrap();

    assert_eq!(
        E::pairing(
            cm,
            s.par_iter()
                .enumerate()
                .map(|(k, &i)| { ggs[n - 1 - i] * (z.pow([k as u64 + 1])) })
                .sum::<E::G2>()
        ),
        E::pairing(cc, gg) + E::pairing(c, ggs[n - 1])
    );
    assert_eq!(d, c * eta + g * b1 + h * b2);
    let [x]: [E::ScalarField; 1] = verifier_state.challenge_scalars().unwrap();

    let t_x = l_x
        .par_iter()
        .zip(&r_x)
        .map(|(&a, b)| a * b)
        .sum::<E::ScalarField>();
    let kk = E::G1::normalize_batch(
        &k.par_iter()
            .enumerate()
            .map(|(i, &k)| k * y.pow([i as u64]).inverse().unwrap())
            .collect::<Vec<_>>(),
    );
    let zeta = (0..k.len())
        .into_par_iter()
        .map(|i| y.pow([i as u64]))
        .sum::<E::ScalarField>()
        * (z - z.square())
        - (0..n)
            .into_par_iter()
            .map(|i| z.pow([i as u64 + 2]))
            .sum::<E::ScalarField>()
            * E::ScalarField::from((1u128 << l) - 1);
    assert_eq!(
        g * t_x + h * tau_x,
        c + g * zeta + tt_1 * x + tt_2 * x.square()
    );
    let p = a
        + b * x
        + j.par_iter().sum::<E::G1>() * -z
        + k.par_iter().sum::<E::G1>() * z
        + E::G1::msm_unchecked(
            &kk,
            &(0..kk.len())
                .into_par_iter()
                .map(|i| {
                    let (j, k) = i.div_rem(&l);
                    E::ScalarField::from(1u128 << k) * z.pow([j as u64 + 1])
                })
                .collect::<Vec<_>>(),
        );
    assert_eq!(
        p,
        E::G1::msm_unchecked(&j, &l_x) + E::G1::msm_unchecked(&kk, &r_x) + h * rho_x
    );
}

#[cfg(test)]
mod tests {
    use ark_std::time::Instant;

    use ark_bn254::Bn254;
    use ark_ec::{PrimeGroup, ScalarMul};
    use rand::Rng;
    use spongefish::{
        codecs::arkworks_algebra::{FieldDomainSeparator, GroupDomainSeparator},
        DomainSeparator,
    };

    use crate::pows;

    use super::*;

    fn test<E: Pairing>(n: usize, l: usize) {
        trait DS<G: CurveGroup> {
            fn add(self) -> Self;
        }

        impl<G> DS<G> for DomainSeparator
        where
            G: CurveGroup,
            Self: GroupDomainSeparator<G> + FieldDomainSeparator<G::ScalarField>,
        {
            fn add(self) -> Self {
                self.add_points(2, "A, B")
                    .challenge_scalars(2, "y, z")
                    .add_points(5, "T1, T2, c, c', D")
                    .challenge_scalars(1, "eta")
                    .add_scalars(2, "b1, b2")
                    .challenge_scalars(1, "x")
            }
        }

        let domain_separator = DomainSeparator::new("test").ratchet();
        let domain_separator = DS::<E::G1>::add(domain_separator);

        let rng = &mut thread_rng();
        let g = E::G1::generator();
        let gg = E::G2::generator();

        let s = E::ScalarField::rand(rng);
        let s_pows = pows(s, n * 2);
        let r = E::ScalarField::rand(rng);

        let gs = g.batch_mul(&s_pows);
        let ggs = gg.batch_mul(&s_pows[1..n + 1]);

        let h = g * r;
        let hs = h.batch_mul(&s_pows[1..n + 1]);

        let v = (0..n)
            .map(|_| rng.gen_range(0..1u128 << l))
            .map(E::ScalarField::from)
            .collect::<Vec<_>>();
        let w = E::ScalarField::rand(rng);
        let cm = E::G1::msm_unchecked(&gs[..n], &v) + h * w;
        let s = (0..n).collect::<Vec<_>>();

        let mut prover_state = domain_separator.to_prover_state();
        prover_state.ratchet().unwrap();

        let now = Instant::now();
        let proof = prove::<E>(&mut prover_state, g, &gs, h, &hs, &v, w, &s, l);
        println!("{} {} {:?}", n, l, now.elapsed());

        let mut verifier_state = domain_separator.to_verifier_state(prover_state.narg_string());
        verifier_state.ratchet().unwrap();
        let now = Instant::now();
        verify::<E>(&mut verifier_state, g, h, gg, &ggs, cm, &s, l, proof);
        println!("{} {} {:?}", n, l, now.elapsed());
    }

    #[test]
    fn test_rp() {
        for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            for l in [8, 16, 32, 64] {
                test::<Bn254>(n, l);
            }
        }
    }
}
