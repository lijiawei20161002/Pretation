from load_balance_gym.envs.param import config

def generate_job(np_random):
    #size = int((np_random.pareto(config.job_size_pareto_shape)+1)*config.job_size_pareto_scale)
    p = np_random.random()
    if p < 0.9:
        size = 1
    else:
        size = 10
    t = int(np_random.exponential(config.job_interval))
    return t, size

def generate_jobs(num_stream_jobs, np_random):
    all_t = []
    all_size = []

    t = 0
    for _ in range(num_stream_jobs):
        dt, size = generate_job(np_random)
        t += dt
        all_t.append(t)
        all_size.append(size)

    return all_t, all_size