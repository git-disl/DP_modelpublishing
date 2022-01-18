"""
This code is used to compare Strong composition, moments accountant, and zCDP with refined composition for disjoint datasets
We compare them in terms of epsilon under (epsilon,delta)-DP
"""

from __future__ import division
import math
import gaussian_moments as gm

#this is CCS16 implementation for adv composition
def compute_epsilon(target_delta, sigma, ratio, iterations):
    delta = target_delta/iterations
    delta = delta/ratio
    epsilon = math.sqrt(2.0 * math.log(1.25 / delta)) / sigma
    #print epsilon, delta
    delta_s = delta*ratio
    epsilon_s = math.log(1.0 + ratio * (math.exp(epsilon) - 1.0))
    #print epsilon_s, delta_s
    return (math.sqrt(epsilon_s**2*iterations),  iterations*delta_s)

def moments_epsilon(target_delta, sigma, ratio, iterations):
    #Gaussian_moments
    max_lmbd = 32
    lmbds = range(1, max_lmbd + 1)
    eps_gm = []
    log_moments = []
    for lmbd in lmbds:
        log_moment = gm.compute_log_moment(ratio, sigma, iterations, lmbd)
        log_moments.append((lmbd, log_moment))
    eps_e, delta_e = gm.get_privacy_spent(log_moments, target_delta=target_delta)
    eps_gm.append(eps_e)
    print(eps_gm)

def zcdptodp_epsilon(target_delta, sigma, ratio, iterations):
    delta_iter=1e-8
    epsilon_iter = math.sqrt(2.0 * math.log(1.25 / delta_iter)) / sigma
    delta_s = ratio*delta_iter
    epsilon_s = math.log(1.0 + ratio * (math.exp(epsilon_iter) - 1.0))
    totaldelta =  iterations*delta_s
    totalepsilon = math.sqrt(iterations*epsilon_s**2)
    lamba= (target_delta-totaldelta)/math.sqrt(math.pi/2)/totalepsilon
    lamba= math.sqrt(-math.log(lamba))
    eps = 1/2*totalepsilon**2+math.sqrt(2)*lamba*totalepsilon
    print(eps)

def compare_accountants(totalexamples, group_size, iterations, sigma, delta):
    itersinepoch = totalexamples//group_size
    epochs = iterations//itersinepoch
    q = float(group_size)/totalexamples
    target_delta = delta
    print((itersinepoch, epochs, q))

    #linear composition
    epsilon_linear = []
    for k in [(x + 1) * itersinepoch for x in range(epochs)]:
        delta_iter = target_delta/(k+1)/q
        epsilon_iter = math.sqrt(2.0 * math.log(1.25 / delta_iter)) / sigma
        delta_iter_s = delta_iter*q  #not used
        epsilon_iter_s = math.log(1.0 + q * (math.exp(epsilon_iter) - 1.0))
        epsilon_linear.append(k*epsilon_iter_s)
    print("epsilon_linear", epsilon_linear)

    #advanced composition
    #each step is (q*epsilon, q*delta)-DP
    #amplified privacy parameters with sampling ratio q
    epsilon_adv = []
    for k in [(x + 1) * itersinepoch for x in range(epochs)]:
        delta_iter = target_delta/((k+1)*q)
        # guassian mechanism
        epsilon_iter = math.ceil(math.sqrt(2.0 * math.log(1.25 / delta_iter))) / sigma
        ##under sampling
        delta_iter_s = q * delta_iter
        epsilon_iter_s = math.log(1.0 + q * (math.exp(epsilon_iter) - 1.0))
        delta_prime=target_delta/(k+1)
        if(delta_prime<=0):
            print('delta_prime is not larger than 0 with %d'%k)
            break
        epsilon_e = epsilon_iter_s*math.sqrt(2.0*k*math.log(1.0/delta_prime))+k*epsilon_iter_s*(math.exp(epsilon_iter_s)-1.0)
        delta_e = k*delta_iter_s+delta_prime #not used
        epsilon_adv.append(epsilon_e)
        #print([epsilon_iter_p*k, delta_iter_p*k])
    print('epsilon_adv', epsilon_adv)

    #dp learning's advance composition implementation
    epsilon_adv = []
    target_delta = delta
    for k in [(x + 1) * itersinepoch for x in range(epochs)]:
        # guassian mechanism
        delta_iter = target_delta/q/k
        epsilon_iter = math.sqrt(2.0 * math.log(1.25 / delta_iter)) / sigma
        ##under sampling
        delta_iter_s = q * delta_iter
        epsilon_iter_s = math.log(1.0 + q * (math.exp(epsilon_iter) - 1.0))
        #print([epsilon_iter_p*k, delta_iter_p*k])
        epsilon_adv.append(math.sqrt(k*epsilon_iter_s*epsilon_iter_s))
    print('epsilon_adv_deep',epsilon_adv)

    #Gaussian_moments
    max_lmbd = 32
    lmbds = range(1, max_lmbd + 1)
    eps_gm = []
    for k in [(x + 1) * itersinepoch for x in range(epochs)]:
        log_moments = []
        for lmbd in lmbds:
            log_moment = gm.compute_log_moment(q, sigma, k, lmbd)
            log_moments.append((lmbd, log_moment))
        eps_e, delta_e = gm.get_privacy_spent(log_moments, target_delta=delta)
        eps_gm.append(eps_e)
    print('epsilon_MA', eps_gm)

    #compute zCDP
    eps_zcdp=[]
    rho_iter = 1.0/(2.0*sigma**2)
    rho_e = rho_iter
    for e in range(1, epochs+1):
        rho_c = rho_e*e
        epsilon_e = rho_c + 2*math.sqrt(rho_c*math.log(1/delta))
        eps_zcdp.append(epsilon_e)
    print('epsilon_zCDP',eps_zcdp)

    # compute zCDP with sampling--approximate bound 1
    eps_zcdp_sampled = []
    rho_iter = 1.0/(2.0*sigma**2)*q**2/(1-q)
    rho_e = rho_iter*itersinepoch
    for e in range(1, epochs+1):
        rho_c = rho_e*e
        epsilon_e = (q**3)*itersinepoch*e +rho_c + 2*math.sqrt(rho_c*(math.log(1.0/delta)))
        eps_zcdp_sampled.append(epsilon_e)
    print('epsilon_sampled_zCDP', eps_zcdp_sampled)

    # compute zCDP with random sampling updated----approximate bound 2
    eps_zcdp_sampled_v2 = []
    rho_iter = 1.0/(sigma**2)*q**2
    rho_e = rho_iter*itersinepoch
    for e in range(1, epochs+1):
        rho_c = rho_e*e
        if(e<10):
            th = 1 / math.exp(rho_c * sigma ** 4 * (math.log(1 / q / sigma)) ** 2)
            if delta < th:
                epsilon_e = rho_c*(sigma**2 *math.log(1/q/sigma)+1)-math.log(delta)/sigma**2/math.log(1/q/sigma)
            else:
                epsilon_e = rho_c + 2 * math.sqrt(rho_c * (math.log(1.0 / delta)))
        else:
            epsilon_e = rho_c + 2*math.sqrt(rho_c*(math.log(1.0/delta)))
        eps_zcdp_sampled_v2.append(epsilon_e)
    print('epsilon_sampled_zCDP_v2', eps_zcdp_sampled_v2)

    return [epsilon_adv, eps_gm, eps_zcdp]

def main():
    totalexamples = 50000
    group_size = 500
    iterations = 40000
    delta = 1e-5
    sigma = 6

    # fxied sampling ratio q=0.01, sigma=10
    moments_epsilon(delta, 10, 0.01, iterations)
    zcdptodp_epsilon(delta, 10, 0.01, iterations)
    print("compute", compute_epsilon(delta, sigma, float(group_size)/totalexamples, iterations))
    compare_accountants(totalexamples, group_size, iterations, sigma, delta)

    # varying sampling ratio
    for i in range(1, 11):
        q = 0.01*i
        print('sample ratio', q)
        compare_accountants(totalexamples, int(totalexamples*q), int(1/q*200), sigma, delta)

    # varying sigma
    for sigma in range(5, 15):
       print('sigma', sigma)
       compare_accountants(totalexamples, group_size, 20000, sigma, delta)

if __name__ == '__main__':
  main()
