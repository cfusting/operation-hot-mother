import argparse
import pandas as pd
import sys
import time

import numpy as np

from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.examples.datasets import HIGGS, load_dataset
from numpyro.infer import HMC, HMCECS, MCMC, NUTS, SVI, Trace_ELBO, autoguide


# def model(data, obs, subsample_size):
#     n, m = data.shape
#     theta = numpyro.sample("theta", dist.Normal(jnp.zeros(m), 0.5 * jnp.ones(m)))
#     with numpyro.plate("N", n, subsample_size=subsample_size):
#         batch_feats = numpyro.subsample(data, event_dim=1)
#         batch_obs = numpyro.subsample(obs, event_dim=0)
#         numpyro.sample(
#             "obs", dist.Bernoulli(logits=theta @ batch_feats.T), obs=batch_obs
#         )


def nonlin(x):
    return jnp.tanh(x)


def model(X, Y, subsample_size=None):

    D_X, D_Y, D_H = X.shape[1], 1, 

    # w1 = numpyro.sample(
    #     "w1", dist.Normal(jnp.zeros(D_X), jnp.ones(D_X))
    # )
    # w1 = numpyro.sample(
    #     "w1", dist.Normal(jnp.zeros((D_X, D_H)), jnp.ones((D_X, D_H)))
    # )
    w2 = numpyro.sample("w2", dist.Normal(jnp.zeros((D_X, D_H)), jnp.ones((D_X, D_H))))
    w3 = numpyro.sample("w3", dist.Normal(jnp.zeros((D_H, D_Y)), jnp.ones((D_H, D_Y))))

    # we put a prior on the observation noise
    # prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    # sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    # observe data
    with numpyro.plate("N", X.shape[0], subsample_size=subsample_size):
        x = numpyro.subsample(X, event_dim=1)
        y = numpyro.subsample(Y, event_dim=0)

        # z1 = nonlin(jnp.matmul(jnp.array(X)[idx], w1))
        z2 = nonlin(jnp.matmul(x, w2))
        z3 = jnp.matmul(z2, w3).flatten()

        # Should be a beta distribution or categorical.
        # Numerai values are 0, .25, .50, .75, 1
        # return numpyro.sample("Y", dist.Normal(w1 @ jnp.array(X)[idx].T, 1), obs=jnp.array(Y)[idx])
        numpyro.sample("Y", dist.Normal(z3, 1), obs=y)


def run_hmcecs(hmcecs_key, args, data, obs, inner_kernel):
    svi_key, mcmc_key = random.split(hmcecs_key)

    # find reference parameters for second order taylor expansion to estimate likelihood (taylor_proxy)
    optimizer = numpyro.optim.Adam(step_size=1e-3)
    guide = autoguide.AutoDelta(model)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    params, losses = svi.run(
        svi_key, args.num_svi_steps, data, obs, args.subsample_size
    )
    ref_params = {
        # "prec_obs": params["prec_obs_auto_loc"],
        # "w1": params["w1_auto_loc"], 
        "w2": params["w2_auto_loc"], 
        "w3": params["w3_auto_loc"],
    }

    # ref_params = {
    #     "theta": params["theta_auto_loc"],
    # }

    # taylor proxy estimates log likelihood (ll) by
    # taylor_expansion(ll, theta_curr) +
    #     sum_{i in subsample} ll_i(theta_curr) - taylor_expansion(ll_i, theta_curr) around ref_params
    proxy = HMCECS.taylor_proxy(ref_params)

    kernel = HMCECS(inner_kernel, num_blocks=args.num_blocks, proxy=proxy)
    mcmc = MCMC(kernel, num_warmup=args.num_warmup, num_samples=args.num_samples)

    mcmc.run(mcmc_key, data, obs, args.subsample_size)
    mcmc.print_summary()
    return losses, mcmc.get_samples()


def run_hmc(mcmc_key, args, data, obs, kernel):
    mcmc = MCMC(kernel, num_warmup=args.num_warmup, num_samples=args.num_samples)
    mcmc.run(mcmc_key, data, obs, None)
    mcmc.print_summary()
    return mcmc.get_samples()


def main(args):
    assert (
        11_000_000 >= args.num_datapoints
    ), "11,000,000 data points in the Higgs dataset"
    # full dataset takes hours for plain hmc!
    # if args.dataset == "higgs":
    #     _, fetch = load_dataset(
    #         HIGGS, shuffle=False, num_datapoints=args.num_datapoints
    #     )
    #     data, obs = fetch()
    # else:
    #     data, obs = (np.random.normal(size=(10, 28)), np.ones(10))

    df = pd.read_csv(args.data_path)
    data = df.iloc[:, 3:30].values
    obs = df.iloc[:, -1].values
    del df

    hmcecs_key, hmc_key = random.split(random.PRNGKey(args.rng_seed))

    # choose inner_kernel
    if args.inner_kernel == "hmc":
        inner_kernel = HMC(model)
    else:
        inner_kernel = NUTS(model)

    start = time.time()
    losses, hmcecs_samples = run_hmcecs(hmcecs_key, args, data, obs, inner_kernel)
    hmcecs_runtime = time.time() - start

    start = time.time()
    hmc_samples = run_hmc(hmc_key, args, data, obs, inner_kernel)
    hmc_runtime = time.time() - start


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Hamiltonian Monte Carlo with Energy Conserving Subsampling"
    )
    parser.add_argument("--subsample_size", type=int, default=1300)
    parser.add_argument("--num_svi_steps", type=int, default=5000)
    parser.add_argument("--num_blocks", type=int, default=100)
    parser.add_argument("--num_warmup", type=int, default=500)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--num_datapoints", type=int, default=1_500_000)
    parser.add_argument(
        "--dataset", type=str, choices=["higgs", "mock"], default="higgs"
    )
    parser.add_argument(
        "--inner_kernel", type=str, choices=["nuts", "hmc"], default="nuts"
    )
    parser.add_argument("--device", default="cpu", type=str, choices=["cpu", "gpu"])
    parser.add_argument(
        "--rng_seed", default=37, type=int, help="random number generator seed"
    )
    parser.add_argument("--data-path", type=str, help="CSV")

    args = parser.parse_args()

    numpyro.set_platform(args.device)

    main(args)
