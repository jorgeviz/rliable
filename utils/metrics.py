import random
import math
from sklearn.metrics.pairwise import cosine_similarity

def get_primes(n_primes, st=2):
    """ Get N prime numbers
    """
    _prims = []
    x = st
    while len(_prims) < n_primes:
        if all([x%y > 0 for y in range(2,x)]):
            _prims.append(x)
        x+=1
    return _prims

def get_prime_larger_than(n):
    """ Get a primer number larger than `num`
        Fast prime implementation from:
        https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n
    """
    sieve = [True] * (n+1)
    for i in range(3,int(n**0.5)+1,2):
        if sieve[i]:
            sieve[i*i::2*i]=[False]*((n-i*i-1)//(2*i)+1)
    return [i for i in range(3,n,2) if sieve[i]][-1]


def get_hash_family(n_funcs, num_shingles):
    """ Get N hash functions:
            f(x = ((a*x + b) % p ) 
    """
    h_funcs = []
    a_done = []
    b_done = []
    p = get_prime_larger_than(num_shingles)
    while h_funcs.__len__() < n_funcs:
        a = random.randint(1, num_shingles)
        b = random.randint(0, num_shingles)
        f_string =  ("lambda x: (({}*x + {}) % {})".format(a,b,p))
        if a in a_done and b in b_done:
            continue
        h_funcs.append(eval(f_string))
        a_done.append(a); b_done.append(b)
    return h_funcs


def jaccard_on_idxs(sx, sy):
    """ Jaccard similarity of 
        sets of indexes of ocurrences
    """
    return len(sx.intersection(sy)) \
        / len(sx.union(sy))

def l2_norm(x):
    """ Compute L2-Norm 
    """
    sq_v = [pow(j,2) for j in x]
    sum_sq = sum(sq_v)
    return math.sqrt(sum_sq)

def pearson_correlation(rt):
    """ Pearson correlation of 2 
        sets of ratings
    """
    # Compute rating means
    m_i, m_j = [], []
    for k,v in rt.items():
        m_i.append(v[0])
        m_j.append(v[1])
    m_i = sum(m_i) / len(m_i)
    m_j = sum(m_j) / len(m_j)
    # Compute dot product, and l2-norms
    dot_rtc, rtc_i, rtc_j = 0, 0, 0
    for k, v in rt.items():
        c_i = v[0] - m_i
        c_j = v[1] - m_j
        dot_rtc += c_i * c_j
        rtc_i += pow(c_i, 2)
        rtc_j += pow(c_j, 2)
    rtc_n = math.sqrt(rtc_i) * math.sqrt(rtc_j)
    if rtc_n == 0:
        return 0
    return dot_rtc / rtc_n

def compute_avg_return(environment, policy, num_episodes=10):
    """ Computing average reward in N episodes

    Parameters
    ----------
    environment: tf_agents.environemnts.tf_py_environment.TFPyEnvironment
        Environment
    policy : tf_agents.policies.Policy
        Agent policy
    num_episodes : int, optional
        Num of episodes to eval, by default 10

    Returns
    -------
    float
        Average reward
    """
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]