
import numpy as np
import scipy
import scipy.stats


def run(data, n_iter = 1000):

    samples = [None for i in range(n_iter)]
    values = [None for i in range(n_iter)]

    samples[0] = np.random.normal(size = 3)
    values[0] = log_like(samples[0], data)

    for i in range(1, n_iter):

        par = np.random.choice(len(samples[i-1]))

        samples[i] = np.copy(samples[i-1])

        samples[i][par] = proposal(samples[i][par])

        old_like = values[i-1]
        new_like = log_like(samples[i], data)

        if np.log(np.random.random()) < new_like - old_like:
            values[i] = new_like
        else:
            samples[i] = np.copy(samples[i-1])
            values[i] = old_like

    final = samples[-1]
    return {'mean0':final[0], 'mean1':final[1], 'sd':np.exp(final[2])}

def proposal(value):
    return value + np.random.normal()*0.1

def log_like(pars, data):
    mu0, mu1, log_sd = pars
    sd = np.exp(log_sd)

    comp0_loglike = np.log(0.5) + scipy.stats.norm.logpdf(data, mu0, sd)
    comp1_loglike = np.log(0.5) + scipy.stats.norm.logpdf(data, mu1, sd)

    return sum(scipy.misc.logsumexp([comp0_loglike, comp1_loglike], axis = 0))


# if __name__ == "__main__":
#
#     pars = [-5,10,0.1]
#
#     data = []
#     for i in range(1000):
#         if np.random.random() < 0.5:
#             data += [np.random.normal()*pars[2] + pars[0]]
#         else:
#             data += [np.random.normal()*pars[2] + pars[1]]
#
#     results = run(data)
#     print results
