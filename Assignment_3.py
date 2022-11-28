# %% [markdown]
# # Setup

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
rng = np.random.default_rng()

# %%
def error(observed, theoretical):
    return abs((theoretical - observed) / observed)

# %%
def mean_over_median(a: np.ndarray, axis=None) -> float:
    return np.mean(a, axis=axis) / np.median(a, axis=axis)

# %%
def jackknife(sample: np.ndarray, estimator: callable) -> np.ndarray:
    return estimator(
        np.vstack(
            tuple(
                np.concatenate((sample[:i], sample[i + 1 :]))
                for i in range(sample.size)
            )
        ),
        axis=-1,
    )

# %%
def bootstrap(
    sample: np.ndarray, estimator: callable, resamples: int = 10_000
) -> np.ndarray:
    return estimator(
        rng.choice(sample, (resamples, sample.size)).reshape((resamples, sample.size)),
        axis=-1,
    )

# %% [markdown]
# # Problem 1

# %% [markdown]
# ## A)
# (Note, the following graphs plot density, not total occurrences)

# %%
alpha, beta, samples = 4, 1, 10**7

gamma = rng.gamma(alpha, beta, samples)
sample = rng.choice(gamma, 100, replace=False)

# %% [markdown]
# ### Hidden Side (Population)

# %%
plt.hist(gamma, bins=1000, density=True)
plt.savefig(fname="Problem 1A Population.png")
plt.clf()

# %% [markdown]
# ### Visible Side (Sample)

# %%
plt.hist(sample, bins=sample.size, density=True)
plt.savefig(fname="Problem 1A Sample.png")
plt.clf()

# %% [markdown]
# ### Sample Mean $\mu$

# %%
np.mean(sample)

# %% [markdown]
# ### Analytic Mean $\alpha \over \beta$

# %%
analytic_mean = alpha / beta
analytic_mean

# %% [markdown]
# ### Sample Variance $\sigma$

# %%
np.std(sample)

# %% [markdown]
# ### Analytic Variance $ \sqrt{\frac{\alpha}{\beta^2}} $

# %%
(alpha / beta**2) ** 0.5

# %% [markdown]
# ## B)

# %% [markdown]
# ### Population Estimator (mean over median)

# %%
gamma_estimator = np.mean(gamma) / np.median(gamma)
gamma_estimator

# %% [markdown]
# ### Sample estimator

# %%
sample_estimator = np.mean(sample) / np.median(sample)
sample_estimator

# %% [markdown]
# #### Sample estimator Error (%)

# %%
sample_error = 100 * error(sample_estimator, gamma_estimator)
sample_error

# %% [markdown]
# ### Bootstrap

# %%
bootstrap_sample = bootstrap(sample, mean_over_median)

# %%
plt.hist(bootstrap_sample, bins=100)
plt.savefig(fname="Problem 1B Bootstrap.png")
plt.clf()


# %% [markdown]
# #### Bootstrap estimator

# %%
bootstrap_estimator = np.mean(bootstrap_sample)
bootstrap_estimator

# %% [markdown]
# #### Bootstrap Error (%)

# %%
bootstrap_error = 100 * error(bootstrap_estimator, gamma_estimator)
bootstrap_error

# %% [markdown]
# ### Summary

# %%
print("---")
print("Estimator: Mean over Median")
print(f"Population: {gamma_estimator:.4}")
print()
print(f"Sample: {sample_estimator:.4}")
print(f"\tError: {sample_error:.4}%")
print()
print(f"Bootstrap: {bootstrap_estimator:.4}")
print(f"\tError: {bootstrap_error:.4}%")
print("---")

# %% [markdown]
# # Problem 2

# %% [markdown]
# ## Jackknife

# %%
jackknife_sample = jackknife(sample, mean_over_median)

# %% [markdown]
# ### Jackknife Histogram

# %%
plt.hist(jackknife_sample, bins=100)
plt.savefig(fname="Problem 2 Jackknife.png")
plt.clf()

# %% [markdown]
# ### Jackknife Estimator

# %%
jackknife_estimator = np.mean(jackknife_sample)
jackknife_estimator

# %% [markdown]
# ### Jackknife Error

# %% [markdown]
# #### Error with Sample

# %%
jackknife_error = 100 * error(jackknife_estimator, gamma_estimator)
jackknife_error

# %% [markdown]
# #### Error with Analytic

# %%
analytic_median = alpha - 1 + np.log(2)

analytic_estimator = analytic_mean / analytic_median
analytic_estimator

# %%
jackknife_error_analytic = 100 * error(jackknife_estimator, analytic_estimator)
jackknife_error_analytic

# %% [markdown]
# ### Summary

# %%
print("---")
print("Estimator: Mean over Median")
print(f"Population: {gamma_estimator:.4}")
print()
print(f"Sample: {sample_estimator:.4}")
print(f"\tError: {sample_error:.4}%")
print()
print(f"Jackknife: {jackknife_estimator:.4}")
print(f"\tError (vs Population): {jackknife_error:.4}%")
print(f"\tError (vs Analytic): {jackknife_error_analytic:.4}%")
print("---")

# %% [markdown]
# # Problem 3

# %%
p3_samples = 10**7
p3_N1 = rng.normal(1, 2, p3_samples)
p3_N2 = rng.normal(4, 1, p3_samples)

mask = rng.random(p3_samples) < 0.3

p3_population = p3_N1 * mask + p3_N2 * np.logical_not(mask)
p3_sample = rng.choice(p3_population, size=100, replace=False)

# %% [markdown]
# ## Population and Sample

# %% [markdown]
# ### Population

# %%
p3_population_estimator = np.mean(p3_population) / np.median(p3_population)

# %%
plt.hist(p3_population, bins=1000, density=True)
plt.savefig(fname="Problem 3 Population.png")
plt.clf()

# %% [markdown]
# ### Sample

# %%
p3_sample_estimator = np.mean(p3_sample) / np.median(p3_sample)

# %%
plt.hist(p3_sample, 100)
plt.savefig(fname="Problem 3 Sample.png")
plt.clf()

# %% [markdown]
# ## Jackknife

# %%
p3_jackknife = jackknife(p3_sample, mean_over_median)
p3_jack_estimator = np.mean(p3_jackknife)

# %%
plt.hist(p3_jackknife, 100)
plt.savefig(fname="Problem 3 Jackknife.png")
plt.clf()

# %% [markdown]
# ## Bootstrap

# %%
p3_bootstrap = bootstrap(p3_sample, estimator=mean_over_median)
p3_boot_estimator = np.mean(p3_bootstrap)

# %%
plt.hist(p3_bootstrap, 100)
plt.savefig(fname="Problem 3 Bootstrap.png")
plt.clf()

# %% [markdown]
# ## Summary

# %%
print("---")
print("Estimator: Mean over Median")
print(f"Population: {p3_population_estimator:.4}")
print()
print(f"Sample: {p3_sample_estimator:.4}")
print(f"\tError: {error(p3_sample_estimator, p3_population_estimator):.4}%")
print()
print(f"Jackknife: {p3_jack_estimator:.4}")
print(f"\tError: {error(p3_jack_estimator, p3_population_estimator):.4}%")
print()
print(f"Bootstrap: {p3_boot_estimator:.4}")
print(f"\tError: {error(p3_boot_estimator, p3_population_estimator):.4}%")
print("---")


