import argparse
import pandas as pd
import os
import time
import sys

import numpy as np

from jax import vmap
import jax
import jax.numpy as jnp
import jax.random as random

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import HMCECS, MCMC, NUTS, SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoBNAFNormal
from numpyro.infer.reparam import NeuTraReparam


def nonlin(x):
    return jnp.tanh(x)


def model(X, Y, D_H, subsample_size=None):

    D_X, D_Y = X.shape[1], 1

    w2 = numpyro.sample("w2", dist.Normal(jnp.zeros((D_X, D_H)), jnp.ones((D_X, D_H))))
    w3 = numpyro.sample("w3", dist.Normal(jnp.zeros((D_H, D_Y)), jnp.ones((D_H, D_Y))))

    # we put a prior on the observation noise
    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    # observe data
    with numpyro.plate("N", X.shape[0], subsample_size=subsample_size):
        x = numpyro.subsample(X, event_dim=1)
        y = numpyro.subsample(Y, event_dim=0)

        z2 = nonlin(jnp.matmul(x, w2))
        z3 = jnp.matmul(z2, w3)

        # Should be a beta distribution or categorical.
        # Numerai values are 0, .25, .50, .75, 1
        return numpyro.sample("Y", dist.Normal(z3, sigma_obs), obs=y)


# helper function for HMC inference
def run_inference(
    model,
    args,
    rng_key,
    rng_key_svi,
    X,
    Y,
    D_H,
    subsample_size,
    svi_iterations,
    num_blocks,
    svi_subsample_size,
):
    start = time.time()
    jax.profiler.start_trace("/tmp/trace")

    guide = AutoBNAFNormal(model, num_flows=3, hidden_factors=[8])
    svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    print("Running SVI.")
    params, losses = svi.run(
        rng_key_svi, svi_iterations, X, Y, D_H, svi_subsample_size
    )

    neutra = NeuTraReparam(guide, params)
    neutra_model = neutra.reparam(model)

    D_X, D_Y = X.shape[1], 1
    neutra_ref_params = {
        "w2": jnp.zeros((D_X, D_H)), 
        "w3": jnp.zeros((D_H, D_Y)),
    }

    # no need to adapt mass matrix if the flow does a good job
    inner_kernel = NUTS(
        neutra_model,
        # init_strategy=init_to_value(values=neutra_ref_params),
        adapt_mass_matrix=False,
    )
    kernel = HMCECS(
        inner_kernel,
        num_blocks=num_blocks,
        # proxy=HMCECS.taylor_proxy(neutra_ref_params)
    )

    mcmc = MCMC(
        kernel,
        args.num_warmup,
        args.num_samples,
        num_chains=args.num_chains,
        progress_bar=True,
    )
    mcmc.run(
        rng_key, X, Y, D_H, subsample_size, extra_fields=("accept_prob",)
    )

    print("Mean accept prob:", jnp.mean(mcmc.get_extra_fields()["accept_prob"]))
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    jax.profiler.stop_trace()
    return mcmc.get_samples()


# helper function for prediction
def predict(model, rng_key, samples, X, D_H):
    # Each call to the model will use a different seed.
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(X=X, Y=None, D_H=D_H)
    return model_trace["Y"]["value"]


def read_data(file_path):
    df = pd.read_csv(file_path)
    # df = df.sample(frac=0.005)
    X = df.iloc[:, 3:30].values
    Y = df.iloc[:, -1].values
    print(f"X shape: {X.shape}")
    return X, Y


def main(args):
    X, Y = read_data(args.data_path)

    assert (
        X.shape[0] >= args.subsample_size
    ), "Number of observations must be larger than subsample size."

    rng_key, rng_key_svi, rng_key_predict = random.split(random.PRNGKey(0), 3)
    samples = run_inference(
        model,
        args,
        rng_key,
        rng_key_svi,
        X,
        Y,
        args.num_hidden,
        args.subsample_size,
        args.svi_iterations,
        args.num_blocks,
        args.svi_subsample_size,
    )

    # predict Y_test at inputs X_test
    # vmap_args = (
    #     samples,
    #     random.split(rng_key_predict, args.num_samples * args.num_chains),
    # )
    # predictions = vmap(
    #     lambda samples, rng_key: predict(model, rng_key, samples, X_test, D_H)
    # )(*vmap_args)
    # predictions = predictions[..., 0]

    # # compute mean prediction and confidence interval around median
    # mean_prediction = jnp.mean(predictions, axis=0)
    # percentiles = np.percentile(predictions, [5.0, 95.0], axis=0)


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.6.0")
    parser = argparse.ArgumentParser(description="Bayesian neural network example")
    parser.add_argument("-n", "--num-samples", nargs="?", default=2000, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=1000, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--num-hidden", nargs="?", default=5, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument("--data-path", type=str, help="CSV")
    parser.add_argument("--subsample-size", type=int, default=1000)
    parser.add_argument("--svi-subsample-size", type=int, default=1000)
    parser.add_argument("--svi-iterations", type=int, default=2000)
    parser.add_argument("--num-blocks", type=int, default=10)
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)
    numpyro.enable_validation()

    main(args)
