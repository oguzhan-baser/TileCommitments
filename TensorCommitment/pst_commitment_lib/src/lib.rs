use ark_ec::{
    pairing::{Pairing, PairingOutput},
    AffineRepr, CurveGroup, PrimeGroup, ScalarMul, VariableBaseMSM,
};
use ark_ff::{
    field_hashers::{DefaultFieldHasher, HashToField},
    Field, PrimeField,
};
use ark_serialize::CanonicalSerialize;
use ark_std::{log2, One, UniformRand, Zero};
use blake2::Blake2s256;
use ip::{
    cfg_multi_pairing, InnerProduct, PointPointInnerProduct, PointScalarInnerProduct,
    ScalarPointInnerProduct, ScalarScalarInnerProduct,
};
use rand::Rng;
use rayon::prelude::*;
use std::{
    marker::PhantomData,
    ops::{Add, Mul, MulAssign},
};

// Python bindings
#[cfg(feature = "python")]
use pyo3::prelude::*;

pub mod ip;
pub mod rp;

fn expand<F: PrimeField>(us: &[F]) -> Vec<F> {
    let mut coeffs = vec![F::one()];
    for u in us {
        coeffs = coeffs
            .into_par_iter()
            .flat_map(|p| vec![p * u.inverse().unwrap(), p * u])
            .collect();
    }
    coeffs
}

fn split_at<T>(mut v: Vec<T>, mid: usize) -> (Vec<T>, Vec<T>) {
    let remainder = v.split_off(mid);
    (v, remainder)
}

pub fn pows<F: PrimeField>(v: F, n: usize) -> Vec<F> {
    let mut c = F::one();
    let mut pows = vec![];
    for _ in 0..n {
        pows.push(c);
        c *= v;
    }
    pows
}

fn div<F: PrimeField>(f: Vec<F>, v: F, i: usize, m: usize, d: usize) -> (Vec<F>, Vec<F>) {
    let mut r = f.clone();
    let l = d.pow(m as u32);
    r.resize(l, F::zero());
    let mut q = vec![F::zero(); l];
    let mut c = d - 1;

    let x_i_d_pos = (0..d)
        .into_par_iter()
        .map(|c| {
            let c = c * d.pow((m - i - 1) as u32);
            (0..d.pow(i as u32))
                .flat_map(|k| {
                    let k = k * d.pow((m - i) as u32);
                    (0..d.pow((m - i - 1) as u32)).map(move |j| k + c + j)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    while c > 0 {
        for (&pos_curr, &pos_next) in x_i_d_pos[c].iter().zip(&x_i_d_pos[c - 1]) {
            let coeff = r[pos_curr];
            r[pos_curr] = F::zero();
            r[pos_next] += v * coeff;
            q[pos_next] = coeff;
        }
        c -= 1;
    }

    (q, r)
}

pub struct Pedersen<E: Pairing> {
    _data: PhantomData<E>,
}

impl<E: Pairing> Pedersen<E> {
    pub fn setup<R: Rng>(rng: &mut R, n: usize) -> (Vec<E::G1Affine>, Vec<E::G2Affine>) {
        let g1 = E::G1::generator();
        let g2 = E::G2::generator();
        let s = E::ScalarField::rand(rng);
        let s_pows = pows(s, n + 1);
        (g1.batch_mul(&s_pows), g2.batch_mul(&s_pows))
    }

    pub fn commit(g_pows: &[E::G1Affine], v: &[E::ScalarField]) -> E::G1 {
        E::G1::msm_unchecked(g_pows, v)
    }

    pub fn verify(g_pows: &[E::G1Affine], v: &[E::ScalarField], c: E::G1) -> bool {
        let c_prime = Self::commit(g_pows, v);
        c == c_prime
    }
}

pub struct AFGHO<E: Pairing> {
    _data: PhantomData<E>,
}

impl<E: Pairing> AFGHO<E> {
    pub fn setup<R: Rng>(rng: &mut R, n: usize) -> (Vec<E::G2Affine>, E::G1Affine) {
        let g1 = E::G1::generator();
        let g2 = E::G2::generator();
        let s = E::ScalarField::rand(rng);
        let s_pows = pows(s, n);
        (g2.batch_mul(&s_pows), (g1 * s).into())
    }

    pub fn commit(g_pows: &[E::G2Affine], v: &[E::G1Affine]) -> PairingOutput<E> {
        cfg_multi_pairing::<E>(v, &g_pows[..v.len()])
    }

    pub fn verify(g_pows: &[E::G2Affine], v: &[E::G1Affine], c: PairingOutput<E>) -> bool {
        let c_prime = Self::commit(g_pows, v);
        c == c_prime
    }
}

pub struct GIPA<IP1: InnerProduct, IP2: InnerProduct<LHS = IP1::LHS>> {
    _data: PhantomData<(IP1, IP2)>,
}

impl<
        IP1: InnerProduct<
            LHS: MulAssign<IP2::RHS> + Add<Output = IP1::LHS> + Zero,
            RHS: MulAssign<IP2::RHS> + Add<Output = IP1::RHS> + Zero,
            Output: Mul<IP2::RHS, Output = IP1::Output> + Add<Output = IP1::Output>,
        >,
        IP2: InnerProduct<
            LHS = IP1::LHS,
            RHS: PrimeField,
            Output: Mul<IP2::RHS, Output = IP2::Output> + Add<Output = IP2::Output>,
        >,
    > GIPA<IP1, IP2>
{
    pub fn prove(
        mut a: Vec<IP1::LHS>,
        mut b: Vec<IP1::RHS>,
        mut c: Vec<IP2::RHS>,
        ab: &IP1::Output,
        ac: &IP2::Output,
    ) -> (
        Vec<IP2::RHS>,
        IP1::LHS,
        IP1::RHS,
        IP2::RHS,
        Vec<(IP1::Output, IP1::Output, IP2::Output, IP2::Output)>,
    ) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), c.len());
        let mut n = a.len();
        assert!(n.is_power_of_two());

        let hasher = <DefaultFieldHasher<Blake2s256> as HashToField<IP2::RHS>>::new(&[]);

        let mut proof = vec![];
        let mut u: [IP2::RHS; 1] = hasher.hash_to_field(&{
            let mut bytes = vec![];
            ab.serialize_compressed(&mut bytes).unwrap();
            ac.serialize_compressed(&mut bytes).unwrap();
            bytes
        });
        let mut us = vec![];
        while n > 1 {
            n >>= 1;
            let (a_l, a_r) = split_at(a, n);
            let (b_l, b_r) = split_at(b, n);
            let (c_l, c_r) = split_at(c, n);
            let l_ab = IP1::inner_product(&a_l, &b_r);
            let r_ab = IP1::inner_product(&a_r, &b_l);
            let l_ac = IP2::inner_product(&a_l, &c_r);
            let r_ac = IP2::inner_product(&a_r, &c_l);
            proof.push((l_ab, r_ab, l_ac, r_ac));
            u = hasher.hash_to_field(&{
                let mut bytes = vec![];
                u[0].serialize_compressed(&mut bytes).unwrap();
                l_ab.serialize_compressed(&mut bytes).unwrap();
                r_ab.serialize_compressed(&mut bytes).unwrap();
                l_ac.serialize_compressed(&mut bytes).unwrap();
                r_ac.serialize_compressed(&mut bytes).unwrap();
                bytes
            });
            us.push(u[0]);
            let u = u[0];
            let v = u.inverse().unwrap();
            a = a_l
                .into_par_iter()
                .zip(a_r.into_par_iter())
                .map(|(mut l, mut r)| {
                    l *= u;
                    r *= v;
                    l + r
                })
                .collect();
            b = b_l
                .into_par_iter()
                .zip(b_r.into_par_iter())
                .map(|(mut l, mut r)| {
                    l *= v;
                    r *= u;
                    l + r
                })
                .collect();
            c = c_l
                .into_par_iter()
                .zip(c_r.into_par_iter())
                .map(|(l, r)| l * v + r * u)
                .collect();
        }

        (us, a[0], b[0], c[0], proof)
    }

    pub fn verify(
        a0: IP1::LHS,
        b0: IP1::RHS,
        c0: IP2::RHS,
        mut ab: IP1::Output,
        mut ac: IP2::Output,
        pi: &[(IP1::Output, IP1::Output, IP2::Output, IP2::Output)],
    ) -> (bool, Vec<IP2::RHS>) {
        let hasher = <DefaultFieldHasher<Blake2s256> as HashToField<IP2::RHS>>::new(&[]);

        let mut u: [IP2::RHS; 1] = hasher.hash_to_field(&{
            let mut bytes = vec![];
            ab.serialize_compressed(&mut bytes).unwrap();
            ac.serialize_compressed(&mut bytes).unwrap();
            bytes
        });
        let mut us = vec![];
        for (l_ab, r_ab, l_ac, r_ac) in pi {
            u = hasher.hash_to_field(&{
                let mut bytes = vec![];
                u[0].serialize_compressed(&mut bytes).unwrap();
                l_ab.serialize_compressed(&mut bytes).unwrap();
                r_ab.serialize_compressed(&mut bytes).unwrap();
                l_ac.serialize_compressed(&mut bytes).unwrap();
                r_ac.serialize_compressed(&mut bytes).unwrap();
                bytes
            });
            us.push(u[0]);
            let u = u[0];
            let uu = u * u;
            let v = u.inverse().unwrap();
            let vv = v * v;
            ab = ab + *r_ab * vv + *l_ab * uu;
            ac = ac + *r_ac * vv + *l_ac * uu;
        }
        (
            IP1::inner_product(&[a0], &[b0]) == ab && IP2::inner_product(&[a0], &[c0]) == ac,
            us,
        )
    }
}

pub struct PST<E: Pairing> {
    _data: PhantomData<E>,
}

impl<E: Pairing> PST<E> {
    pub fn setup<R: Rng>(rng: &mut R, m: usize, d: usize) -> (Vec<E::G1Affine>, Vec<E::G2Affine>) {
        let g1 = E::G1::generator();
        let g2 = E::G2::generator();
        let ts = (0..m)
            .map(|_| E::ScalarField::rand(rng))
            .collect::<Vec<_>>();
        let mut ts_pows = vec![E::ScalarField::one()];
        for t in &ts {
            let t_pows = pows(*t, d);
            ts_pows = ts_pows
                .into_par_iter()
                .flat_map(|p| t_pows.iter().map(move |t| p * t).collect::<Vec<_>>())
                .collect();
        }
        let ck = g1.batch_mul(&ts_pows);
        let vk = g2.batch_mul(&ts);
        (ck, vk)
    }

    pub fn commit(ck: &[E::G1Affine], f: &[E::ScalarField]) -> E::G1 {
        E::G1::msm_unchecked(ck, f)
    }

    pub fn prove(
        d: usize,
        ck: &[E::G1Affine],
        v: &[E::ScalarField],
        f: &[E::ScalarField],
    ) -> Vec<E::G1> {
        let mut pi = vec![];
        let mut f = f.to_vec();
        for i in 0..v.len() {
            let (q, r) = div(f, v[i], i, v.len(), d);
            pi.push(E::G1::msm_unchecked(ck, &q));
            f = r;
        }
        pi
    }

    pub fn verify(
        vk: &[E::G2Affine],
        v: &[E::ScalarField],
        c: E::G1,
        e: E::ScalarField,
        pi: &[E::G1],
    ) -> bool {
        let mut l = pi.to_vec();
        let mut r = vk
            .par_iter()
            .zip(v)
            .map(|(vk, v)| E::G2::from(*vk) - E::G2::generator() * v)
            .collect::<Vec<_>>();
        l.push(E::G1::generator() * e - c);
        r.push(E::G2::generator());

        cfg_multi_pairing::<E>(&E::G1::normalize_batch(&l), &E::G2::normalize_batch(&r)).is_zero()
    }
}

pub struct MMP<E: Pairing> {
    _data: PhantomData<E>,
}

impl<E: Pairing> MMP<E> {
    pub fn setup<R: Rng>(
        rng: &mut R,
        m: usize,
        n: usize,
        d: usize,
    ) -> (
        (Vec<E::G1Affine>, Vec<E::G2Affine>, Vec<E::G1Affine>),
        (Vec<E::G2Affine>, E::G1Affine, Vec<E::G2Affine>),
    ) {
        let (ck_pst, vk_pst) = PST::<E>::setup(rng, m, d);
        let (ck_ped, vk_ped) = Pedersen::<E>::setup(rng, n);
        let (ck_afg, vk_afg) = AFGHO::<E>::setup(rng, n);
        ((ck_ped, ck_afg, ck_pst), (vk_ped, vk_afg, vk_pst))
    }

    pub fn commit(
        (_ck_ped, ck_afg, ck_pst): &(Vec<E::G1Affine>, Vec<E::G2Affine>, Vec<E::G1Affine>),
        fs: &[Vec<E::ScalarField>],
    ) -> (Vec<E::G1>, PairingOutput<E>) {
        let c_pst = fs
            .par_iter()
            .map(|f| PST::<E>::commit(ck_pst, f))
            .collect::<Vec<_>>();
        let c_afg = AFGHO::<E>::commit(ck_afg, &E::G1::normalize_batch(&c_pst));
        (c_pst, c_afg)
    }

    pub fn prove(
        d: usize,
        (ck_ped, ck_afg, ck_pst): &(Vec<E::G1Affine>, Vec<E::G2Affine>, Vec<E::G1Affine>),
        fs: &[Vec<E::ScalarField>],
        c_pst: &[E::G1],
        c_afg: &PairingOutput<E>,
        v: &[E::ScalarField],
    ) -> (
        E::G1,
        (
            Vec<E::G1>,
            (
                E::G1,
                E::G2,
                E::ScalarField,
                E::G2,
                Vec<(PairingOutput<E>, PairingOutput<E>, E::G1, E::G1)>,
            ),
            (
                E::ScalarField,
                E::G1,
                E::ScalarField,
                E::G1,
                Vec<(E::G1, E::G1, E::ScalarField, E::ScalarField)>,
            ),
            E::G1,
            E::ScalarField,
        ),
    ) {
        let es = fs
            .par_iter()
            .map(|f| {
                let mut ts_pows = vec![E::ScalarField::one()];
                for t in v {
                    let t_pows = pows(*t, d);
                    ts_pows = ts_pows
                        .into_iter()
                        .flat_map(|p| t_pows.iter().map(move |t| p * t))
                        .collect();
                }
                f.iter()
                    .zip(&ts_pows)
                    .map(|(fi, ti)| *fi * ti)
                    .sum::<E::ScalarField>()
            })
            .collect::<Vec<_>>();
        let c_ped = Pedersen::<E>::commit(ck_ped, &es);
        let hasher = <DefaultFieldHasher<Blake2s256> as HashToField<E::ScalarField>>::new(&[]);

        let r: [E::ScalarField; 1] = hasher.hash_to_field(&{
            let mut bytes = vec![];
            c_afg.serialize_compressed(&mut bytes).unwrap();
            v.serialize_compressed(&mut bytes).unwrap();
            c_ped.serialize_compressed(&mut bytes).unwrap();
            bytes
        });
        let r_pows = pows(r[0], fs.len());
        let mut fr = vec![E::ScalarField::zero(); fs[0].len()];
        for (f, r) in fs.iter().zip(&r_pows) {
            for i in 0..f.len() {
                fr[i] += f[i] * r;
            }
        }
        let cr = E::G1::msm_unchecked(&E::G1::normalize_batch(c_pst), &r_pows);
        let er = es
            .par_iter()
            .zip(&r_pows)
            .map(|(e, r)| *e * r)
            .sum::<E::ScalarField>();
        let pi_1 = PST::<E>::prove(d, ck_pst, v, &fr);
        let (us_2, a0_2, b0_2, c0_2, pi_ipa_2) =
            GIPA::<PointPointInnerProduct<E>, PointScalarInnerProduct<E::G1>>::prove(
                c_pst.to_vec(),
                ck_afg.par_iter().map(|g| E::G2::from(*g)).collect(),
                r_pows.clone(),
                c_afg,
                &cr,
            );
        let pi_kzg_2 = {
            let r: [E::ScalarField; 1] = hasher.hash_to_field(&{
                let mut bytes = vec![];
                pi_ipa_2.serialize_compressed(&mut bytes).unwrap();
                bytes
            });
            let mut g = expand(&us_2);
            let mut h = vec![E::ScalarField::zero(); g.len() - 1];
            while g.len() > 1 {
                let cur_q_coeff = g.pop().unwrap();
                let cur_q_degree = g.len() - 1;
                h[cur_q_degree] = cur_q_coeff;
                g[cur_q_degree] += cur_q_coeff * r[0];
            }
            E::G2::msm_unchecked(ck_afg, &h)
        };

        let (us_3, a0_3, b0_3, c0_3, pi_ipa_3) = GIPA::<
            ScalarPointInnerProduct<E::G1>,
            ScalarScalarInnerProduct<E::ScalarField>,
        >::prove(
            es,
            ck_ped[..ck_ped.len() - 1]
                .par_iter()
                .map(|g| E::G1::from(*g))
                .collect(),
            r_pows,
            &c_ped,
            &er,
        );
        let pi_kzg_3 = {
            let r: [E::ScalarField; 1] = hasher.hash_to_field(&{
                let mut bytes = vec![];
                pi_ipa_3.serialize_compressed(&mut bytes).unwrap();
                bytes
            });
            let mut g = expand(&us_3);
            let mut h = vec![E::ScalarField::zero(); g.len() - 1];
            while g.len() > 1 {
                let cur_q_coeff = g.pop().unwrap();
                let cur_q_degree = g.len() - 1;
                h[cur_q_degree] = cur_q_coeff;
                g[cur_q_degree] += cur_q_coeff * r[0];
            }
            E::G1::msm_unchecked(ck_ped, &h)
        };

        (
            c_ped,
            (
                pi_1,
                (a0_2, b0_2, c0_2, pi_kzg_2, pi_ipa_2),
                (a0_3, b0_3, c0_3, pi_kzg_3, pi_ipa_3),
                cr,
                er,
            ),
        )
    }

    fn verify_ipa2(
        vk_kzg: E::G1Affine,
        c_afg: PairingOutput<E>,
        mut r: E::ScalarField,
        cr: E::G1,
        (a0, b0, c0, pi_kzg, pi_ipa): &(
            E::G1,
            E::G2,
            E::ScalarField,
            E::G2,
            Vec<(PairingOutput<E>, PairingOutput<E>, E::G1, E::G1)>,
        ),
    ) -> bool {
        let (pi_ok, us) = GIPA::<PointPointInnerProduct<E>, PointScalarInnerProduct<E::G1>>::verify(
            *a0, *b0, *c0, c_afg, cr, pi_ipa,
        );
        let b0_ok = {
            let hasher = <DefaultFieldHasher<Blake2s256> as HashToField<E::ScalarField>>::new(&[]);

            let r: [E::ScalarField; 1] = hasher.hash_to_field(&{
                let mut bytes = vec![];
                pi_ipa.serialize_compressed(&mut bytes).unwrap();
                bytes
            });
            let mut r_pow = r[0];
            let mut e = E::ScalarField::one();
            for u in us.iter().rev() {
                e *= u.inverse().unwrap() + *u * r_pow;
                r_pow *= r_pow;
            }
            cfg_multi_pairing::<E>(
                &E::G1::normalize_batch(&[
                    E::G1::from(vk_kzg) - E::G1::generator() * r[0],
                    -E::G1::generator(),
                ]),
                &E::G2::normalize_batch(&[*pi_kzg, *b0 - E::G2::generator() * e]),
            )
            .is_zero()
        };

        let c0_ok = {
            let mut e = E::ScalarField::one();
            for u in us.iter().rev() {
                e *= u.inverse().unwrap() + *u * r;
                r *= r;
            }
            e
        } == *c0;

        pi_ok && b0_ok && c0_ok
    }

    fn verify_ipa3(
        vk_kzg: E::G2Affine,
        c_ped: E::G1,
        mut r: E::ScalarField,
        er: E::ScalarField,
        (a0, b0, c0, pi_kzg, pi_ipa): &(
            E::ScalarField,
            E::G1,
            E::ScalarField,
            E::G1,
            Vec<(E::G1, E::G1, E::ScalarField, E::ScalarField)>,
        ),
    ) -> bool {
        let (pi_ok, us) = GIPA::<
            ScalarPointInnerProduct<E::G1>,
            ScalarScalarInnerProduct<E::ScalarField>,
        >::verify(*a0, *b0, *c0, c_ped, er, pi_ipa);
        let b0_ok = {
            let hasher = <DefaultFieldHasher<Blake2s256> as HashToField<E::ScalarField>>::new(&[]);

            let r: [E::ScalarField; 1] = hasher.hash_to_field(&{
                let mut bytes = vec![];
                pi_ipa.serialize_compressed(&mut bytes).unwrap();
                bytes
            });
            let mut r_pow = r[0];
            let mut e = E::ScalarField::one();
            for u in us.iter().rev() {
                e *= u.inverse().unwrap() + *u * r_pow;
                r_pow *= r_pow;
            }
            cfg_multi_pairing::<E>(
                &E::G1::normalize_batch(&[*pi_kzg, *b0 - E::G1::generator() * e]),
                &E::G2::normalize_batch(&[
                    E::G2::from(vk_kzg) - E::G2::generator() * r[0],
                    -E::G2::generator(),
                ]),
            )
            .is_zero()
        };

        let c0_ok = {
            let mut e = E::ScalarField::one();
            for u in us.iter().rev() {
                e *= u.inverse().unwrap() + *u * r;
                r *= r;
            }
            e
        } == *c0;

        pi_ok && b0_ok && c0_ok
    }

    pub fn verify(
        (vk_ped, vk_afg, vk_pst): &(Vec<E::G2Affine>, E::G1Affine, Vec<E::G2Affine>),
        c_afg: PairingOutput<E>,
        c_ped: E::G1,
        v: &[E::ScalarField],
        (pi_1, pi_2, pi_3, cr, er): &(
            Vec<E::G1>,
            (
                E::G1,
                E::G2,
                E::ScalarField,
                E::G2,
                Vec<(PairingOutput<E>, PairingOutput<E>, E::G1, E::G1)>,
            ),
            (
                E::ScalarField,
                E::G1,
                E::ScalarField,
                E::G1,
                Vec<(E::G1, E::G1, E::ScalarField, E::ScalarField)>,
            ),
            E::G1,
            E::ScalarField,
        ),
    ) -> bool {
        let hasher = <DefaultFieldHasher<Blake2s256> as HashToField<E::ScalarField>>::new(&[]);

        let r: [E::ScalarField; 1] = hasher.hash_to_field(&{
            let mut bytes = vec![];
            c_afg.serialize_compressed(&mut bytes).unwrap();
            v.serialize_compressed(&mut bytes).unwrap();
            c_ped.serialize_compressed(&mut bytes).unwrap();
            bytes
        });
        PST::<E>::verify(vk_pst, v, *cr, *er, pi_1)
            && Self::verify_ipa2(*vk_afg, c_afg, r[0], *cr, pi_2)
            && Self::verify_ipa3(vk_ped[1], c_ped, r[0], *er, pi_3)
    }

    pub fn proof_size(m: usize, n: usize) -> usize {
        E::G1::zero().compressed_size() * m
            + (E::G1::zero().compressed_size()
                + E::G2::zero().compressed_size()
                + E::ScalarField::zero().compressed_size()
                + E::G2::zero().compressed_size()
                + (PairingOutput::<E>::zero().compressed_size()
                    + PairingOutput::<E>::zero().compressed_size()
                    + E::G1::zero().compressed_size()
                    + E::G1::zero().compressed_size())
                    * log2(n) as usize)
            + (E::ScalarField::zero().compressed_size()
                + E::G1::zero().compressed_size()
                + E::ScalarField::zero().compressed_size()
                + E::G1::zero().compressed_size()
                + (E::G1::zero().compressed_size()
                    + E::G1::zero().compressed_size()
                    + E::ScalarField::zero().compressed_size()
                    + E::ScalarField::zero().compressed_size())
                    * log2(n) as usize)
            + E::G1::zero().compressed_size()
            + E::ScalarField::zero().compressed_size()
    }
}

pub struct ZKMMP<E: Pairing> {
    _data: PhantomData<E>,
}

impl<E: Pairing> ZKMMP<E> {
    pub fn setup<R: Rng>(
        rng: &mut R,
        m: usize,
        n: usize,
        d: usize,
    ) -> (
        (Vec<E::G1Affine>, Vec<E::G2Affine>, Vec<E::G1Affine>),
        (Vec<E::G2Affine>, E::G1Affine, Vec<E::G2Affine>),
        E::G1,
        E::G2,
        Vec<E::G1Affine>,
    ) {
        let (ck_pst, vk_pst) = PST::<E>::setup(rng, m, d);
        let (ck_ped, vk_ped) = Pedersen::<E>::setup(rng, n * 2);
        let (ck_afg, vk_afg) = AFGHO::<E>::setup(rng, n);

        let zeta = E::ScalarField::rand(rng);
        let h1 = E::G1::generator() * zeta;
        let h2 = E::G2::generator() * zeta;
        (
            (ck_ped.clone(), ck_afg, ck_pst),
            (vk_ped, vk_afg, vk_pst),
            h1,
            h2,
            ck_ped
                .into_iter()
                .map(|g| (g * zeta).into_affine())
                .collect::<Vec<_>>(),
        )
    }

    pub fn commit<R: Rng>(
        rng: &mut R,
        (_ck_ped, ck_afg, ck_pst): &(Vec<E::G1Affine>, Vec<E::G2Affine>, Vec<E::G1Affine>),
        h1: E::G1,
        fs: &[Vec<E::ScalarField>],
    ) -> (Vec<E::G1>, PairingOutput<E>, Vec<E::ScalarField>) {
        let rhos = fs
            .iter()
            .map(|_| E::ScalarField::rand(rng))
            .collect::<Vec<_>>();
        let c_pst = fs
            .par_iter()
            .zip(&rhos)
            .map(|(f, rho)| PST::<E>::commit(ck_pst, f) + h1 * rho)
            .collect::<Vec<_>>();
        let c_afg = AFGHO::<E>::commit(ck_afg, &E::G1::normalize_batch(&c_pst));
        (c_pst, c_afg, rhos)
    }

    pub fn prove<R: Rng>(
        rng: &mut R,
        m: usize,
        d: usize,
        (ck_ped, ck_afg, ck_pst): &(Vec<E::G1Affine>, Vec<E::G2Affine>, Vec<E::G1Affine>),
        h1: E::G1,
        hs: &[E::G1Affine],
        fs: &[Vec<E::ScalarField>],
        c_pst: &[E::G1],
        c_afg: &PairingOutput<E>,
        v: &[E::ScalarField],
        rhos: &[E::ScalarField],
    ) -> (
        E::G1,
        (
            (
                E::G1,
                E::G2,
                E::ScalarField,
                E::G2,
                Vec<(PairingOutput<E>, PairingOutput<E>, E::G1, E::G1)>,
            ),
            (E::G1, E::G1, E::G1, E::ScalarField, E::ScalarField),
            Vec<E::G1>,
            E::G1,
            E::G1,
        ),
    ) {
        let rhoo = E::ScalarField::rand(rng);
        let es = fs
            .par_iter()
            .map(|f| {
                let mut ts_pows = vec![E::ScalarField::one()];
                for t in v {
                    let t_pows = pows(*t, d);
                    ts_pows = ts_pows
                        .into_iter()
                        .flat_map(|p| t_pows.iter().map(move |t| p * t))
                        .collect();
                }
                f.iter()
                    .zip(&ts_pows)
                    .map(|(fi, ti)| *fi * ti)
                    .sum::<E::ScalarField>()
            })
            .collect::<Vec<_>>();
        let c_ped = Pedersen::<E>::commit(ck_ped, &es) + h1 * rhoo;
        let hasher = <DefaultFieldHasher<Blake2s256> as HashToField<E::ScalarField>>::new(&[]);

        let r: [E::ScalarField; 1] = hasher.hash_to_field(&{
            let mut bytes = vec![];
            c_afg.serialize_compressed(&mut bytes).unwrap();
            v.serialize_compressed(&mut bytes).unwrap();
            c_ped.serialize_compressed(&mut bytes).unwrap();
            bytes
        });
        let r_pows = pows(r[0], fs.len());

        let cr = E::G1::msm_unchecked(&E::G1::normalize_batch(c_pst), &r_pows);

        let (us_2, a0_2, b0_2, c0_2, pi_ipa_2) =
            GIPA::<PointPointInnerProduct<E>, PointScalarInnerProduct<E::G1>>::prove(
                c_pst.to_vec(),
                ck_afg.par_iter().map(|g| E::G2::from(*g)).collect(),
                r_pows.clone(),
                c_afg,
                &cr,
            );
        let pi_kzg_2 = {
            let r: [E::ScalarField; 1] = hasher.hash_to_field(&{
                let mut bytes = vec![];
                pi_ipa_2.serialize_compressed(&mut bytes).unwrap();
                bytes
            });
            let mut g = expand(&us_2);
            let mut h = vec![E::ScalarField::zero(); g.len() - 1];
            while g.len() > 1 {
                let cur_q_coeff = g.pop().unwrap();
                let cur_q_degree = g.len() - 1;
                h[cur_q_degree] = cur_q_coeff;
                g[cur_q_degree] += cur_q_coeff * r[0];
            }
            E::G2::msm_unchecked(ck_afg, &h)
        };

        let varrho = E::ScalarField::rand(rng);
        let t = es
            .iter()
            .zip(&r_pows)
            .map(|(e, r)| *r * e)
            .sum::<E::ScalarField>();
        let c_v = E::G1::generator() * t + h1 * varrho;

        let n = fs.len();

        let mut x = vec![E::ScalarField::zero(); n * 2];
        for i in 1..=n {
            for j in 0..n {
                x[i + j] += es[j] * r_pows[n - i];
            }
        }
        x[n] = E::ScalarField::zero();

        let cc = E::G1::msm_unchecked(ck_ped, &x)
            + E::G1::msm_unchecked(
                &hs[1..],
                &r_pows.iter().rev().map(|r| rhoo * r).collect::<Vec<_>>(),
            )
            - hs[n] * varrho;
        let varrho_1 = E::ScalarField::rand(rng);
        let varrho_2 = E::ScalarField::rand(rng);
        let dd = E::G1::generator() * varrho_1 + h1 * varrho_2;
        let eta: [E::ScalarField; 1] = hasher.hash_to_field(&{
            let mut bytes = vec![];
            r.serialize_compressed(&mut bytes).unwrap();
            c_v.serialize_compressed(&mut bytes).unwrap();
            cc.serialize_compressed(&mut bytes).unwrap();
            dd.serialize_compressed(&mut bytes).unwrap();
            bytes
        });
        let eta = eta[0];
        let b1 = varrho_1 - eta * t;
        let b2 = varrho_2 - eta * varrho;

        let mus = (0..m)
            .map(|_| E::ScalarField::rand(rng))
            .collect::<Vec<_>>();
        let mut fs = fs.to_vec();
        let l = d.pow(m as u32);
        let pi_pst = (0..m)
            .map(|j| {
                let qq = fs
                    .par_iter_mut()
                    .zip(&r_pows)
                    .map(|(f, rr)| {
                        let (mut q, r) = div(f.clone(), v[j], j, v.len(), d);
                        *f = r;
                        for v in q.iter_mut() {
                            *v *= rr;
                        }
                        q
                    })
                    .reduce(
                        || vec![E::ScalarField::zero(); l],
                        |a, b| a.into_iter().zip(b).map(|(x, y)| x + y).collect(),
                    );
                E::G1::msm_unchecked(ck_pst, &qq) + h1 * mus[j]
            })
            .collect::<Vec<_>>();

        let theta = E::G1::generator()
            * (mus
                .par_iter()
                .zip(v)
                .map(|(m, v)| *m * v)
                .sum::<E::ScalarField>()
                + rhos
                    .par_iter()
                    .zip(r_pows)
                    .map(|(rho, r)| *rho * r)
                    .sum::<E::ScalarField>()
                - varrho)
            - E::G1::msm_unchecked(
                &(0..m)
                    .map(|i| ck_pst[d.pow((m - 1 - i) as u32)])
                    .collect::<Vec<_>>(),
                &mus,
            );

        (
            c_ped,
            (
                (a0_2, b0_2, c0_2, pi_kzg_2, pi_ipa_2),
                (c_v, cc, dd, b1, b2),
                pi_pst,
                cr,
                theta,
            ),
        )
    }

    fn verify_ipa2(
        vk_kzg: E::G1Affine,
        c_afg: PairingOutput<E>,
        mut r: E::ScalarField,
        cr: E::G1,
        (a0, b0, c0, pi_kzg, pi_ipa): &(
            E::G1,
            E::G2,
            E::ScalarField,
            E::G2,
            Vec<(PairingOutput<E>, PairingOutput<E>, E::G1, E::G1)>,
        ),
    ) -> bool {
        let (pi_ok, us) = GIPA::<PointPointInnerProduct<E>, PointScalarInnerProduct<E::G1>>::verify(
            *a0, *b0, *c0, c_afg, cr, pi_ipa,
        );
        let b0_ok = {
            let hasher = <DefaultFieldHasher<Blake2s256> as HashToField<E::ScalarField>>::new(&[]);

            let r: [E::ScalarField; 1] = hasher.hash_to_field(&{
                let mut bytes = vec![];
                pi_ipa.serialize_compressed(&mut bytes).unwrap();
                bytes
            });
            let mut r_pow = r[0];
            let mut e = E::ScalarField::one();
            for u in us.iter().rev() {
                e *= u.inverse().unwrap() + *u * r_pow;
                r_pow *= r_pow;
            }
            cfg_multi_pairing::<E>(
                &E::G1::normalize_batch(&[
                    E::G1::from(vk_kzg) - E::G1::generator() * r[0],
                    -E::G1::generator(),
                ]),
                &E::G2::normalize_batch(&[*pi_kzg, *b0 - E::G2::generator() * e]),
            )
            .is_zero()
        };

        let c0_ok = {
            let mut e = E::ScalarField::one();
            for u in us.iter().rev() {
                e *= u.inverse().unwrap() + *u * r;
                r *= r;
            }
            e
        } == *c0;

        pi_ok && b0_ok && c0_ok
    }

    pub fn verify(
        n: usize,
        (vk_ped, vk_afg, vk_pst): &(Vec<E::G2Affine>, E::G1Affine, Vec<E::G2Affine>),
        c_afg: PairingOutput<E>,
        c_ped: E::G1,
        h1: E::G1,
        h2: E::G2,
        v: &[E::ScalarField],
        (pi_2, (c_v, cc, dd, b1, b2), pi_pst, cr, theta): &(
            (
                E::G1,
                E::G2,
                E::ScalarField,
                E::G2,
                Vec<(PairingOutput<E>, PairingOutput<E>, E::G1, E::G1)>,
            ),
            (E::G1, E::G1, E::G1, E::ScalarField, E::ScalarField),
            Vec<E::G1>,
            E::G1,
            E::G1,
        ),
    ) -> bool {
        let hasher = <DefaultFieldHasher<Blake2s256> as HashToField<E::ScalarField>>::new(&[]);

        let r: [E::ScalarField; 1] = hasher.hash_to_field(&{
            let mut bytes = vec![];
            c_afg.serialize_compressed(&mut bytes).unwrap();
            v.serialize_compressed(&mut bytes).unwrap();
            c_ped.serialize_compressed(&mut bytes).unwrap();
            bytes
        });
        let eta: [E::ScalarField; 1] = hasher.hash_to_field(&{
            let mut bytes = vec![];
            r.serialize_compressed(&mut bytes).unwrap();
            c_v.serialize_compressed(&mut bytes).unwrap();
            cc.serialize_compressed(&mut bytes).unwrap();
            dd.serialize_compressed(&mut bytes).unwrap();
            bytes
        });
        let r_pows = pows(r[0], n);

        Self::verify_ipa2(*vk_afg, c_afg, r[0], *cr, pi_2)
            && E::G1::generator() * b1 + h1 * b2 + *c_v * eta[0] == *dd
            && cfg_multi_pairing::<E>(
                &E::G1::normalize_batch(&[-c_ped, *cc, *c_v]),
                &[
                    E::G2::msm_unchecked(
                        &vk_ped[1..=n],
                        &r_pows.into_iter().rev().collect::<Vec<_>>(),
                    )
                    .into_affine(),
                    E::G2Affine::generator(),
                    vk_ped[n],
                ],
            )
            .is_zero()
            && E::pairing(*cr - c_v, E::G2::generator())
                == cfg_multi_pairing::<E>(
                    &E::G1::normalize_batch(pi_pst),
                    &E::G2::normalize_batch(
                        &vk_pst
                            .iter()
                            .zip(v)
                            .map(|(g, v)| *g - E::G2::generator() * v)
                            .collect::<Vec<_>>(),
                    ),
                ) + E::pairing(theta, h2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::{Bn254, Fr, G1Projective, G2Projective};
    use num_integer::binomial;
    use rand::thread_rng;

    #[test]
    fn test_expand() {
        let rng = &mut thread_rng();
        let n = 6;
        let us = (0..n).map(|_| Fr::rand(rng)).collect::<Vec<_>>();
        let f = expand(&us);

        let r = Fr::rand(rng);
        let r_pows = pows(r, 1 << n);
        let f_r = f.iter().zip(&r_pows).map(|(fi, ri)| fi * ri).sum::<Fr>();

        let e = us
            .iter()
            .enumerate()
            .map(|(i, u)| u.inverse().unwrap() + *u * r_pows[1 << (us.len() - 1 - i)])
            .product::<Fr>();

        assert_eq!(f_r, e);
    }

    #[test]
    fn test_pedersen() {
        let mut rng = thread_rng();
        let n = 4;
        let (g_pows, _) = Pedersen::<Bn254>::setup(&mut rng, n);
        let v = (0..n).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
        let c = Pedersen::<Bn254>::commit(&g_pows, &v);
        assert!(Pedersen::<Bn254>::verify(&g_pows, &v, c));
    }

    #[test]
    fn test_afgho() {
        let mut rng = thread_rng();
        let n = 4;
        let (g_pows, _) = AFGHO::<Bn254>::setup(&mut rng, n);
        let v = (0..n)
            .map(|_| G1Projective::rand(&mut rng).into_affine())
            .collect::<Vec<_>>();
        let c = AFGHO::<Bn254>::commit(&g_pows, &v);
        assert!(AFGHO::<Bn254>::verify(&g_pows, &v, c));
    }

    #[test]
    fn test_gipa() {
        let mut rng = thread_rng();
        let n = 8;
        let a = (0..n)
            .map(|_| G1Projective::rand(&mut rng))
            .collect::<Vec<_>>();
        let b = (0..n)
            .map(|_| G2Projective::rand(&mut rng))
            .collect::<Vec<_>>();
        let c = (0..n).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
        let ab = PointPointInnerProduct::<Bn254>::inner_product(&a, &b);
        let ac = PointScalarInnerProduct::<G1Projective>::inner_product(&a, &c);
        let (_, a0, b0, c0, pi) = GIPA::<
            PointPointInnerProduct<Bn254>,
            PointScalarInnerProduct<G1Projective>,
        >::prove(a.clone(), b.clone(), c.clone(), &ab, &ac);
        let (res, _) =
            GIPA::<PointPointInnerProduct<Bn254>, PointScalarInnerProduct<G1Projective>>::verify(
                a0, b0, c0, ab, ac, &pi,
            );
        assert!(res);
    }

    #[test]
    fn test_pst() {
        let rng = &mut thread_rng();
        let m = 4;
        let d = 3;
        let (ck, vk) = PST::<Bn254>::setup(rng, m, d);
        let f = (0..d.pow(m as u32))
            .map(|_| Fr::rand(rng))
            .collect::<Vec<_>>();
        let v = (0..m).map(|_| Fr::rand(rng)).collect::<Vec<_>>();
        let e = {
            let mut ts_pows = vec![Fr::one()];
            for t in &v {
                let t_pows = pows(*t, d);
                ts_pows = ts_pows
                    .into_iter()
                    .flat_map(|p| t_pows.iter().map(move |t| p * t))
                    .collect();
            }
            f.iter().zip(&ts_pows).map(|(fi, ti)| *fi * ti).sum::<Fr>()
        };
        let c = PST::<Bn254>::commit(&ck, &f);
        let pi = PST::<Bn254>::prove(d, &ck, &v, &f);
        assert!(PST::<Bn254>::verify(&vk, &v, c, e, &pi));
    }

    #[test]
    fn test_mmp() {
        let rng = &mut thread_rng();
        let d = 4;
        let m = 5;
        let n = 16;
        let (ck, vk) = MMP::<Bn254>::setup(rng, m, n, d);
        let fs = (0..n)
            .map(|_| {
                (0..d.pow(m as u32))
                    .map(|_| Fr::rand(rng))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let v = (0..m).map(|_| Fr::rand(rng)).collect::<Vec<_>>();
        let (c_pst, c_afg) = MMP::<Bn254>::commit(&ck, &fs);
        let (c_ped, pi) = MMP::<Bn254>::prove(d, &ck, &fs, &c_pst, &c_afg, &v);
        assert!(MMP::<Bn254>::verify(&vk, c_afg, c_ped, &v, &pi));
    }

    #[test]
    fn test_zkmmp() {
        let rng = &mut thread_rng();
        let d = 2;
        let m = 2;
        let n = 1;
        let (ck, vk, h1, h2, hs) = ZKMMP::<Bn254>::setup(rng, m, n, d);
        let fs = (0..n)
            .map(|_| {
                (0..binomial(m + d - 1, d - 1))
                    .map(|_| Fr::rand(rng))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let v = (0..m).map(|_| Fr::rand(rng)).collect::<Vec<_>>();
        let (c_pst, c_afg, rhos) = ZKMMP::<Bn254>::commit(rng, &ck, h1, &fs);
        let (c_ped, pi) =
            ZKMMP::<Bn254>::prove(rng, m, d, &ck, h1, &hs, &fs, &c_pst, &c_afg, &v, &rhos);
        assert!(ZKMMP::<Bn254>::verify(
            n, &vk, c_afg, c_ped, h1, h2, &v, &pi
        ));
    }

    #[test]
    fn test_proof_size() {
        let rng = &mut thread_rng();
        let d = 4;
        let m = 5;
        let n = 16;
        let (ck, _vk) = MMP::<Bn254>::setup(rng, m, n, d);
        let fs = (0..n)
            .map(|_| {
                (0..d.pow(m as u32))
                    .map(|_| Fr::rand(rng))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let v = (0..m).map(|_| Fr::rand(rng)).collect::<Vec<_>>();
        let (c_pst, c_afg) = MMP::<Bn254>::commit(&ck, &fs);
        let (_c_ped, pi) = MMP::<Bn254>::prove(d, &ck, &fs, &c_pst, &c_afg, &v);
        assert_eq!(
            MMP::<Bn254>::proof_size(m, n),
            pi.0.compressed_size() - 8 + pi.1.compressed_size() - 8 + pi.2.compressed_size() - 8
                + pi.3.compressed_size()
                + pi.4.compressed_size()
        );
    }

    #[test]
    fn test_div() {
        // F(3) * x0^0 * x1^0 * x2^0 + F(2) * x0^0 * x1^0 * x2^1 + F(4) * x0^0 * x1^0 * x2^2 +
        // F(1) * x0^0 * x1^1 * x2^0 + F(3) * x0^0 * x1^1 * x2^1 + F(2) * x0^0 * x1^1 * x2^2 +
        // F(2) * x0^0 * x1^2 * x2^0 + F(4) * x0^0 * x1^2 * x2^1 + F(1) * x0^0 * x1^2 * x2^2 +
        // F(4) * x0^1 * x1^0 * x2^0 + F(1) * x0^1 * x1^0 * x2^1 + F(3) * x0^1 * x1^0 * x2^2 +
        // F(3) * x0^1 * x1^1 * x2^0 + F(2) * x0^1 * x1^1 * x2^1 + F(4) * x0^1 * x1^1 * x2^2 +
        // F(1) * x0^1 * x1^2 * x2^0 + F(3) * x0^1 * x1^2 * x2^1 + F(2) * x0^1 * x1^2 * x2^2 +
        // F(2) * x0^2 * x1^0 * x2^0 + F(4) * x0^2 * x1^0 * x2^1 + F(1) * x0^2 * x1^0 * x2^2 +
        // F(4) * x0^2 * x1^1 * x2^0 + F(1) * x0^2 * x1^1 * x2^1 + F(3) * x0^2 * x1^1 * x2^2 +
        // F(3) * x0^2 * x1^2 * x2^0 + F(2) * x0^2 * x1^2 * x2^1 + F(4) * x0^2 * x1^2 * x2^2
        let f = [
            3, 2, 4, 1, 3, 2, 2, 4, 1, 4, 1, 3, 3, 2, 4, 1, 3, 2, 2, 4, 1, 4, 1, 3, 3, 2, 4,
        ]
        .map(Fr::from);
        let v = Fr::from(3);
        let (q, r) = div(f.to_vec(), v, 0, 3, 3);
        for i in 0..q.len() {
            if !q[i].is_zero() {
                print!("{}*x0^{}*x1^{}*x2^{} + ", q[i], i / 9, (i / 3) % 3, i % 3);
            }
        }
        println!();
        println!();
        println!();
        for i in 0..r.len() {
            if !r[i].is_zero() {
                print!("{}*x0^{}*x1^{}*x2^{} + ", r[i], i / 9, (i / 3) % 3, i % 3);
            }
        }
        println!();
    }
}

// Python bindings
#[cfg(feature = "python")]
mod python_bindings {
    use super::*;
    use ark_bn254::{Bn254, Fr, G1Affine, G1Projective as G1, G2Affine};
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    use ark_std::test_rng;
    use num_bigint::{BigInt, BigUint};
    use num_integer::Integer;
    use pyo3::exceptions::PyValueError;
    use pyo3::types::{PyAny, PyList, PyModule};
    use pyo3::{IntoPy, PyObject, PyResult, Python};
    use std::sync::OnceLock;

    fn ark_bigint_to_bigint(value: <Fr as PrimeField>::BigInt) -> BigInt {
        let mut bytes = Vec::with_capacity(value.0.len() * 8);
        for limb in value.0 {
            bytes.extend_from_slice(&limb.to_le_bytes());
        }
        BigInt::from(BigUint::from_bytes_le(&bytes))
    }

    fn fr_modulus() -> &'static BigInt {
        static MODULUS: OnceLock<BigInt> = OnceLock::new();
        MODULUS.get_or_init(|| ark_bigint_to_bigint(Fr::MODULUS))
    }

    fn py_any_to_bigint(value: &PyAny) -> PyResult<BigInt> {
        value.extract::<BigInt>()
    }

    fn py_any_to_fr(value: &PyAny) -> PyResult<Fr> {
        let modulus = fr_modulus();
        let integer = py_any_to_bigint(value)?;
        let reduced = integer.mod_floor(modulus);
        let magnitude = reduced
            .to_biguint()
            .ok_or_else(|| PyValueError::new_err("Expected a finite integer value"))?;
        let bytes = magnitude.to_bytes_le();
        Ok(Fr::from_le_bytes_mod_order(&bytes))
    }

    fn py_list_to_fr_vec(list: &PyList) -> PyResult<Vec<Fr>> {
        list.iter().map(|item| py_any_to_fr(item)).collect()
    }

    fn fr_to_bigint(value: Fr) -> BigInt {
        ark_bigint_to_bigint(value.into_bigint())
    }

    /// Python wrapper for tensor commitment functionality
    #[pyclass]
    pub struct TensorCommitmentWrapper {
        commitment_key: Vec<G1Affine>,
        verification_key: Vec<G2Affine>,
        degree_bound: usize,
        num_variables: usize,
    }

    #[pymethods]
    impl TensorCommitmentWrapper {
        #[new]
        fn new(num_variables: usize, degree_bound: usize) -> Self {
            let rng = &mut test_rng();
            let (ck, vk) = PST::<Bn254>::setup(rng, num_variables, degree_bound);

            TensorCommitmentWrapper {
                commitment_key: ck,
                verification_key: vk,
                degree_bound,
                num_variables,
            }
        }

        /// Commit to polynomial coefficients
        fn commit(&self, coefficients: &PyList) -> PyResult<String> {
            let coeffs = py_list_to_fr_vec(coefficients)?;

            let commitment = PST::<Bn254>::commit(&self.commitment_key, &coeffs);

            // Serialize commitment to hex string for Python
            let mut bytes = Vec::new();
            commitment.serialize_compressed(&mut bytes).unwrap();
            Ok(hex::encode(bytes))
        }

        /// Prove polynomial evaluation at given point
        fn prove(
            &self,
            coefficients: &PyList,
            evaluation_point: &PyList,
            _claimed_evaluation: &PyAny,
        ) -> PyResult<Vec<String>> {
            let coeffs = py_list_to_fr_vec(coefficients)?;
            let point = py_list_to_fr_vec(evaluation_point)?;

            let proof =
                PST::<Bn254>::prove(self.degree_bound, &self.commitment_key, &point, &coeffs);

            // Serialize proof elements to hex strings
            let mut proof_hex = Vec::new();
            for pi in proof {
                let mut bytes = Vec::new();
                pi.serialize_compressed(&mut bytes).unwrap();
                proof_hex.push(hex::encode(bytes));
            }

            Ok(proof_hex)
        }

        /// Verify polynomial evaluation proof
        fn verify(
            &self,
            commitment_hex: &str,
            evaluation_point: &PyList,
            claimed_evaluation: &PyAny,
            proof_hex: &PyList,
        ) -> PyResult<bool> {
            // Deserialize commitment
            let commitment_bytes = hex::decode(commitment_hex).unwrap();
            let commitment = G1::deserialize_compressed(&commitment_bytes[..]).unwrap();

            // Deserialize proof
            let mut proof = Vec::new();
            for item in proof_hex.iter() {
                let hex_str: &str = item.extract()?;
                let bytes = hex::decode(hex_str).unwrap();
                let pi = G1::deserialize_compressed(&bytes[..]).unwrap();
                proof.push(pi);
            }

            // Parse evaluation point
            let point = py_list_to_fr_vec(evaluation_point)?;

            let claimed_eval = py_any_to_fr(claimed_evaluation)?;

            Ok(PST::<Bn254>::verify(
                &self.verification_key,
                &point,
                commitment,
                claimed_eval,
                &proof,
            ))
        }

        /// Get polynomial evaluation at a point (for testing)
        fn evaluate_polynomial(
            &self,
            py: Python,
            coefficients: &PyList,
            evaluation_point: &PyList,
        ) -> PyResult<PyObject> {
            let coeffs = py_list_to_fr_vec(coefficients)?;
            let point = py_list_to_fr_vec(evaluation_point)?;

            // Compute polynomial evaluation
            let mut ts_pows = vec![Fr::one()];
            for t in &point {
                let t_pows = pows(*t, self.degree_bound);
                ts_pows = ts_pows
                    .into_iter()
                    .flat_map(|p| t_pows.iter().map(move |t| p * t))
                    .collect();
            }

            let evaluation = coeffs
                .iter()
                .zip(&ts_pows)
                .map(|(fi, ti)| *fi * ti)
                .sum::<Fr>();

            // Return as Python int with arbitrary precision
            let eval_bigint = fr_to_bigint(evaluation);
            Ok(eval_bigint.into_py(py))
        }
    }

    /// Python module definition
    #[pymodule]
    fn tensorcommitments(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_class::<TensorCommitmentWrapper>()?;
        Ok(())
    }
}
