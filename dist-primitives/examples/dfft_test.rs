use ark_bls12_377::Fr;
use ark_ff::{FftField, PrimeField};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use dist_primitives::{
    channel::channel::MpcSerNet,
    dfft::dfft::{d_fft, fft_in_place_rearrange, d_ifft},
    utils::pack::transpose,
    Opt,
};
use mpc_net::{MpcMultiNet as Net, MpcNet};
use secret_sharing::pss::PackedSharingParams;
use structopt::StructOpt;

pub fn d_fft_test<F: FftField + PrimeField>(
    pp: &PackedSharingParams<F>,
    dom: &Radix2EvaluationDomain<F>,
) {
    let mut rng = ark_std::test_rng();
    let mbyl: usize = dom.size() / pp.l;
    // We apply FFT on this vector
    
    let mut x_coeffs: Vec<F> = Vec::new();
    for i in 0..dom.size() {
        x_coeffs.push(F::from(i as u64));
    }
    let x_evals = dom.fft(&x_coeffs);

    let mut x_coeff_shares = x_coeffs.clone();
    fft_in_place_rearrange(&mut x_coeff_shares);
    
    let mut x_eval_shares = x_evals.clone();
    fft_in_place_rearrange(&mut x_eval_shares);

    // packed coeffs
    let mut pcoeff: Vec<Vec<F>> = Vec::new();
    for i in 0..mbyl {
        pcoeff.push(x_coeff_shares.iter().skip(i).step_by(mbyl).cloned().collect::<Vec<_>>());
        pp.pack_from_public_in_place(&mut pcoeff[i]);
    }

    let pcoeff_share = pcoeff
        .iter()
        .map(|shares| shares[Net::party_id()])
        .collect::<Vec<_>>();

    // packed evals
    let mut peval: Vec<Vec<F>> = Vec::new();
    for i in 0..mbyl {
        peval.push(x_eval_shares.iter().skip(i).step_by(mbyl).cloned().collect::<Vec<_>>());
        pp.pack_from_public_in_place(&mut peval[i]);
    }

    let peval_share = peval
        .iter()
        .map(|shares| shares[Net::party_id()])
        .collect::<Vec<_>>();
    
    let fft_share = d_fft(pcoeff_share, false, 1, false, dom, pp);
    let ifft_share = d_ifft(peval_share, false, 1, false, dom, pp);

    // Send to king who reconstructs and checks the answer
    Net::send_to_king(&fft_share).map(|fft_share| {
        let fft_share = transpose(fft_share);

        let pevals: Vec<F> = fft_share
            .into_iter()
            .flat_map(|shares| pp.unpack(&shares))
            .collect();
        // pevals.reverse(); // todo: implement such that we avoid this reverse

        if Net::am_king() {
            assert_eq!(x_evals, pevals);
        }
    });

    Net::send_to_king(&ifft_share).map(|ifft_share| {
        let ifft_share = transpose(ifft_share);

        let pcoeffs: Vec<F> = ifft_share
            .into_iter()
            .flat_map(|shares| pp.unpack(&shares))
            .collect();
        // pcoeffs.reverse(); // todo: implement such that we avoid this reverse

        if Net::am_king() {
            assert_eq!(x_coeffs, pcoeffs);
        }
    });
}

pub fn main() {
    env_logger::builder().format_timestamp(None).init();

    let opt = Opt::from_args();

    Net::init_from_file(opt.input.to_str().unwrap(), opt.id);
    let pp = PackedSharingParams::<Fr>::new(opt.l);
    let dom = Radix2EvaluationDomain::<Fr>::new(opt.m).unwrap();
    debug_assert_eq!(
        dom.size(),
        opt.m,
        "Failed to obtain domain of size {}",
        opt.m
    );
    d_fft_test::<ark_bls12_377::Fr>(&pp, &dom);

    Net::deinit();
}
