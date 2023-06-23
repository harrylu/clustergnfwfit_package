import emcee
import numpy as np
import corner
import matplotlib.pyplot as plt

filename = 'emcee_backend_2nd_fit_6k_5k.h5'
reader = emcee.backends.HDFBackend(filename)
ndim = 32

try:
    tau = reader.get_autocorr_time()
except Exception as e:
    print(e)
burnin = 2000#int(2 * np.max(tau))
thin = 250#int(0.5 * np.min(tau))
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
#samples[:, [1, 2]] = np.log10(-samples[:, [1, 2]])
# good_index = np.ndarray.flatten(np.argwhere(samples[:, 1] > -30))
# print(good_index.shape)
# samples = samples[good_index]
# print(samples.shape)

log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin).astype(float)
# log_prob_samples = log_prob_samples[good_index]
# log_prior_samples = log_prior_samples[good_index]


print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
print("flat log prob shape: {0}".format(log_prob_samples.shape))
print("flat log prior shape: {0}".format(log_prior_samples.shape))

all_samples = np.concatenate(
    (samples, log_prob_samples[:, None], log_prior_samples[:, None]), axis=1
)

# pnames = ['theta', 'cbrt_p0_90', 'cbrt_p0_150', 'r_x', 'r_y', 'offset_x', 'offset_y', 'c_90', 'c_150', 'cbrt_p0_bolocam', 'c_bolocam']
pnames = ['cbrt_p0_90', 'cbrt_p0_150', 'cbrt_p0_bolocam', 'c_90', 'c_150', 'c_bolocam']
labels = list(pnames)
labels += ["log prob", "log prior"]

# range = [1, ]
figure = corner.corner(all_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True, range=[1 for _ in range(len(labels))])
plt.show()