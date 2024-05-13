import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from einops import repeat
from jax.random import split
from tqdm.auto import tqdm

import util
from mdps import smdp, dsmdp, csmdp, mdsmdp, hsmdp
from mdps.natural_mdps import DiscretePendulum
from mdps.wrappers import LogWrapper
from mdps.wrappers_mine import TimeLimit, FlattenObservationWrapper


axis2name = {
    "sd": "State Dimension",
    "ad": "Action Dimension",
    "od": "Observation Dimension",

    "isv": "Init State Variance",

    "tc": "Transition Complexity",
    "tloc": "Transition Locality",
    "ts": "Transition Stochasticity",

    "oc": "Observation Complexity",
    "os": "Observation Stochasticity",

    "rc": "Reward Complexity",
    "rs": "Reward Stochasticity",
    "rspr": "Reward Sparse", # boolean
    "rsprp": "Reward Sparsity Probability",

    "dc": "Done Complexity",
    "ds": "Done Stochasticity",
    "dsprp": "Done Sparsity Probability",
    
    "tl": "Time Limit",
}

axis2possible_vals = {k: list(range(5)) for k in axis2name}
axis2possible_vals['rspr'] = [0, 1]

family2axes = {
    "mchain": ["sd",       "od", "isv",               "ts",       "os",                                    "ds", "dsprp", "tl"],
    "bandit": ["sd", "ad", "od", "isv",               "ts",       "os",       "rs", "rspr", "rsprp",                          ],
    "dsmdp":  ["sd", "ad", "od", "isv",               "ts",       "os",       "rs", "rspr", "rsprp",       "ds", "dsprp", "tl"],
    "mdsmdp": ["sd", "ad", "od", "isv", "tc",         "ts", "oc", "os", "rc", "rs", "rspr", "rsprp", "dc", "ds", "dsprp", "tl"],
    "csmdp":  ["sd", "ad", "od", "isv", "tc", "tloc", "ts", "oc", "os", "rc", "rs", "rspr", "rsprp", "dc", "ds", "dsprp", "tl"],
    "ucsmdp": ["sd", "ad", "od", "isv", "tc", "tloc", "ts", "oc", "os", "rc", "rs", "rspr", "rsprp", "dc", "ds", "dsprp", "tl"],
    "ecsmdp": ["sd", "ad", "od", "isv", "tc", "tloc", "ts", "oc", "os", "rc", "rs", "rspr", "rsprp", "dc", "ds", "dsprp", "tl"],
    "hsmdp":  ["sd", "ad", "od", "isv", "tc", "tloc", "ts", "oc", "os", "rc", "rs", "rspr", "rsprp", "dc", "ds", "dsprp", "tl"],
}

def create_smdp(env_id, log_wrapper=False):
    env_cfg = dict([sub.split('=') for sub in env_id.split(';')])

    if env_cfg['name'] == 'mchain':
        sd = [4, 16, 64, 256, 1024][int(env_cfg['sd'])]
        # ad = [1, 2, 3, 5, 8][int(env_cfg['ad'])]
        od = [2, 4, 8, 16, 32][int(env_cfg['od'])]
        isv = [.01, .1, .3, 1, 100][int(env_cfg['isv'])]
        # tc = [0, 1, 2, 4, 8][int(env_cfg['tc'])]
        # tloc = [0.01, 0.03, 0.1, 0.3, 1.0][int(env_cfg['tloc'])]
        ts = [.01, .1, .3, 1, 100][int(env_cfg['ts'])]
        # oc = [0, 1, 2, 4, 8][int(env_cfg['oc'])]
        os = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['os'])]
        # rc = [0, 1, 2, 4, 8][int(env_cfg['rc'])]
        # rs = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['rs'])]
        # rspr = [False, True][int(env_cfg['rspr'])]
        # rsprp = [0.01, 0.03, 0.1, 0.3, 0.9][int(env_cfg['rsprp'])]
        # dc = [0, 1, 2, 4, 8][int(env_cfg['dc'])]
        ds = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['ds'])]
        dsprp = [0.0, 0.01, 0.03, 0.1, 0.3][int(env_cfg['dsprp'])]
        tl = [1, 4, 16, 64, 128][int(env_cfg['tl'])]
        model_init = dsmdp.Init(n=sd, temperature=isv)
        model_trans = dsmdp.Transition(n=sd, n_acts=1, temperature=ts)
        model_obs = dsmdp.Observation(n=sd, d_obs=od, std=os)
        model_rew = dsmdp.Reward(n=sd, std=0., sparse=True, sparse_prob=0.)
        model_done = dsmdp.Done(n=sd, std=ds, sparse_prob=dsprp)
        env = smdp.SyntheticMDP(model_init, model_trans, model_obs, model_rew, model_done)
        env = TimeLimit(env, n_steps_max=tl)
    elif env_cfg['name'] == 'bandit':
        sd = [4, 16, 64, 256, 1024][int(env_cfg['sd'])]
        ad = [1, 2, 3, 5, 8][int(env_cfg['ad'])]
        od = [2, 4, 8, 16, 32][int(env_cfg['od'])]
        isv = [.01, .1, .3, 1, 100][int(env_cfg['isv'])]
        # tc = [0, 1, 2, 4, 8][int(env_cfg['tc'])]
        # tloc = [0.01, 0.03, 0.1, 0.3, 1.0][int(env_cfg['tloc'])]
        ts = [.01, .1, .3, 1, 100][int(env_cfg['ts'])]
        # oc = [0, 1, 2, 4, 8][int(env_cfg['oc'])]
        os = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['os'])]
        # rc = [0, 1, 2, 4, 8][int(env_cfg['rc'])]
        rs = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['rs'])]
        rspr = [False, True][int(env_cfg['rspr'])]
        rsprp = [0.01, 0.03, 0.1, 0.3, 0.9][int(env_cfg['rsprp'])]
        # dc = [0, 1, 2, 4, 8][int(env_cfg['dc'])]
        # ds = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['ds'])]
        # dsprp = [0.0, 0.01, 0.03, 0.1, 0.3][int(env_cfg['dsprp'])]
        # tl = [1, 4, 16, 64, 128][int(env_cfg['tl'])]
        model_init = dsmdp.Init(n=sd, temperature=isv)
        model_trans = dsmdp.Transition(n=sd, n_acts=ad, temperature=ts)
        model_obs = dsmdp.Observation(n=sd, d_obs=od, std=os)
        model_rew = dsmdp.Reward(n=sd, std=rs, sparse=rspr, sparse_prob=rsprp)
        model_done = smdp.NeverDone()
        env = smdp.SyntheticMDP(model_init, model_trans, model_obs, model_rew, model_done)
        env = TimeLimit(env, n_steps_max=1)
    elif env_cfg['name'] == 'dsmdp':
        sd = [4, 16, 64, 256, 1024][int(env_cfg['sd'])]
        ad = [1, 2, 3, 5, 8][int(env_cfg['ad'])]
        od = [2, 4, 8, 16, 32][int(env_cfg['od'])]
        isv = [.01, .1, .3, 1, 100][int(env_cfg['isv'])]
        # tc = [0, 1, 2, 4, 8][int(env_cfg['tc'])]
        # tloc = [0.01, 0.03, 0.1, 0.3, 1.0][int(env_cfg['tloc'])]
        ts = [.01, .1, .3, 1, 100][int(env_cfg['ts'])]
        # oc = [0, 1, 2, 4, 8][int(env_cfg['oc'])]
        os = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['os'])]
        # rc = [0, 1, 2, 4, 8][int(env_cfg['rc'])]
        rs = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['rs'])]
        rspr = [False, True][int(env_cfg['rspr'])]
        rsprp = [0.01, 0.03, 0.1, 0.3, 0.9][int(env_cfg['rsprp'])]
        # dc = [0, 1, 2, 4, 8][int(env_cfg['dc'])]
        ds = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['ds'])]
        dsprp = [0.0, 0.01, 0.03, 0.1, 0.3][int(env_cfg['dsprp'])]
        tl = [1, 4, 16, 64, 128][int(env_cfg['tl'])]
        model_init = dsmdp.Init(n=sd, temperature=isv)
        model_trans = dsmdp.Transition(n=sd, n_acts=ad, temperature=ts)
        model_obs = dsmdp.Observation(n=sd, d_obs=od, std=os)
        model_rew = dsmdp.Reward(n=sd, std=rs, sparse=rspr, sparse_prob=rsprp)
        model_done = dsmdp.Done(n=sd, std=ds, sparse_prob=dsprp)
        env = smdp.SyntheticMDP(model_init, model_trans, model_obs, model_rew, model_done)
        env = TimeLimit(env, n_steps_max=tl)
    elif env_cfg['name'] == 'mdsmdp':
        sd = [2, 4, 6, 8, 16][int(env_cfg['sd'])]
        ad = [1, 2, 3, 5, 8][int(env_cfg['ad'])]
        od = [2, 4, 8, 16, 32][int(env_cfg['od'])]
        isv = [.01, .1, .3, 1, 100][int(env_cfg['isv'])]
        tc = [0, 1, 2, 4, 8][int(env_cfg['tc'])]
        # tloc = [0.01, 0.03, 0.1, 0.3, 1.0][int(env_cfg['tloc'])]
        ts = [.01, .1, .3, 1, 100][int(env_cfg['ts'])]
        oc = [0, 1, 2, 4, 8][int(env_cfg['oc'])]
        os = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['os'])]
        rc = [0, 1, 2, 4, 8][int(env_cfg['rc'])]
        rs = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['rs'])]
        rspr = [False, True][int(env_cfg['rspr'])]
        rsprp = [0.01, 0.03, 0.1, 0.3, 0.9][int(env_cfg['rsprp'])]
        dc = [0, 1, 2, 4, 8][int(env_cfg['dc'])]
        ds = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['ds'])]
        dsprp = [0.0, 0.01, 0.03, 0.1, 0.3][int(env_cfg['dsprp'])]
        tl = [1, 4, 16, 64, 128][int(env_cfg['tl'])]
        model_init = mdsmdp.Init(m=sd, n=sd, temperature=isv)
        model_trans = mdsmdp.Transition(m=sd, n=sd, n_acts=ad, temperature=ts, n_layers=tc, d_hidden=16, activation=jax.nn.gelu)
        model_obs = mdsmdp.Observation(m=sd, n=sd, d_obs=od, std=os, n_layers=oc, d_hidden=16, activation=jax.nn.gelu)
        model_rew = mdsmdp.Reward(m=sd, n=sd, std=rs, sparse=rspr, sparse_prob=rsprp, n_layers=rc, d_hidden=16, activation=jax.nn.gelu)
        model_done = mdsmdp.Done(m=sd, n=sd, std=ds, sparse_prob=dsprp, n_layers=dc, d_hidden=16, activation=jax.nn.gelu)
        env = smdp.SyntheticMDP(model_init, model_trans, model_obs, model_rew, model_done)
        env = TimeLimit(env, n_steps_max=tl)
    elif env_cfg['name'] == 'csmdp':
        sd = [2, 4, 8, 16, 32][int(env_cfg['sd'])]
        ad = [1, 2, 3, 5, 8][int(env_cfg['ad'])]
        od = [2, 4, 8, 16, 32][int(env_cfg['od'])]
        isv = [0, 0.1, 0.3, 1.0, 3.0][int(env_cfg['isv'])]
        tc = [0, 1, 2, 4, 8][int(env_cfg['tc'])]
        tloc = [0.01, 0.03, 0.1, 0.3, 1.0][int(env_cfg['tloc'])]
        ts = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['ts'])]
        oc = [0, 1, 2, 4, 8][int(env_cfg['oc'])]
        os = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['os'])]
        rc = [0, 1, 2, 4, 8][int(env_cfg['rc'])]
        rs = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['rs'])]
        rspr = [False, True][int(env_cfg['rspr'])]
        rsprp = [0.01, 0.03, 0.1, 0.3, 0.9][int(env_cfg['rsprp'])]
        dc = [0, 1, 2, 4, 8][int(env_cfg['dc'])]
        ds = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['ds'])]
        dsprp = [0.0, 0.01, 0.03, 0.1, 0.3][int(env_cfg['dsprp'])]
        tl = [1, 4, 16, 64, 128][int(env_cfg['tl'])]
        model_init = csmdp.Init(d_state=sd, std=isv, constraint="clip", n_embeds=None)
        model_trans = csmdp.Transition(d_state=sd, n_acts=ad, std=ts, locality=tloc, constraint='clip', n_layers=tc, d_hidden=16, activation=jax.nn.gelu)
        model_obs = csmdp.Observation(d_state=sd, d_obs=od, std=os, n_layers=oc, d_hidden=16, activation=jax.nn.gelu)
        model_rew = csmdp.Reward(d_state=sd, std=rs, sparse=rspr, sparse_prob=rsprp, n_layers=rc, d_hidden=16, activation=jax.nn.gelu)
        model_done = csmdp.Done(d_state=sd, std=ds, sparse_prob=dsprp, n_layers=dc, d_hidden=16, activation=jax.nn.gelu)
        env = smdp.SyntheticMDP(model_init, model_trans, model_obs, model_rew, model_done)
        env = TimeLimit(env, n_steps_max=tl)
    elif env_cfg['name'] == 'ucsmdp': # all same as csmdp except constraint
        sd = [2, 4, 8, 16, 32][int(env_cfg['sd'])]
        ad = [1, 2, 3, 5, 8][int(env_cfg['ad'])]
        od = [2, 4, 8, 16, 32][int(env_cfg['od'])]
        isv = [0, 0.1, 0.3, 1.0, 3.0][int(env_cfg['isv'])]
        tc = [0, 1, 2, 4, 8][int(env_cfg['tc'])]
        tloc = [0.01, 0.03, 0.1, 0.3, 1.0][int(env_cfg['tloc'])]
        ts = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['ts'])]
        oc = [0, 1, 2, 4, 8][int(env_cfg['oc'])]
        os = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['os'])]
        rc = [0, 1, 2, 4, 8][int(env_cfg['rc'])]
        rs = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['rs'])]
        rspr = [False, True][int(env_cfg['rspr'])]
        rsprp = [0.01, 0.03, 0.1, 0.3, 0.9][int(env_cfg['rsprp'])]
        dc = [0, 1, 2, 4, 8][int(env_cfg['dc'])]
        ds = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['ds'])]
        dsprp = [0.0, 0.01, 0.03, 0.1, 0.3][int(env_cfg['dsprp'])]
        tl = [1, 4, 16, 64, 128][int(env_cfg['tl'])]
        model_init = csmdp.Init(d_state=sd, std=isv, constraint="unit_norm", n_embeds=None)
        model_trans = csmdp.Transition(d_state=sd, n_acts=ad, std=ts, locality=tloc, constraint='unit_norm', n_layers=tc, d_hidden=16, activation=jax.nn.gelu)
        model_obs = csmdp.Observation(d_state=sd, d_obs=od, std=os, n_layers=oc, d_hidden=16, activation=jax.nn.gelu)
        model_rew = csmdp.Reward(d_state=sd, std=rs, sparse=rspr, sparse_prob=rsprp, n_layers=rc, d_hidden=16, activation=jax.nn.gelu)
        model_done = csmdp.Done(d_state=sd, std=ds, sparse_prob=dsprp, n_layers=dc, d_hidden=16, activation=jax.nn.gelu)
        env = smdp.SyntheticMDP(model_init, model_trans, model_obs, model_rew, model_done)
        env = TimeLimit(env, n_steps_max=tl)
    elif env_cfg['name'] == 'ecsmdp': # all same as csmdp except constraint and embeddings
        sd = [2, 4, 8, 16, 32][int(env_cfg['sd'])]
        sde = [16, 64, 256, 1024, 4096][int(env_cfg['sd'])]
        ad = [1, 2, 3, 5, 8][int(env_cfg['ad'])]
        od = [2, 4, 8, 16, 32][int(env_cfg['od'])]
        isv = [0, 0.1, 0.3, 1.0, 3.0][int(env_cfg['isv'])]
        tc = [0, 1, 2, 4, 8][int(env_cfg['tc'])]
        tloc = [0.01, 0.03, 0.1, 0.3, 1.0][int(env_cfg['tloc'])]
        ts = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['ts'])]
        oc = [0, 1, 2, 4, 8][int(env_cfg['oc'])]
        os = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['os'])]
        rc = [0, 1, 2, 4, 8][int(env_cfg['rc'])]
        rs = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['rs'])]
        rspr = [False, True][int(env_cfg['rspr'])]
        rsprp = [0.01, 0.03, 0.1, 0.3, 0.9][int(env_cfg['rsprp'])]
        dc = [0, 1, 2, 4, 8][int(env_cfg['dc'])]
        ds = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['ds'])]
        dsprp = [0.0, 0.01, 0.03, 0.1, 0.3][int(env_cfg['dsprp'])]
        tl = [1, 4, 16, 64, 128][int(env_cfg['tl'])]
        model_init = csmdp.Init(d_state=sd, std=isv, constraint="embeddings", n_embeds=sde)
        model_trans = csmdp.Transition(d_state=sd, n_acts=ad, std=ts, locality=tloc, constraint='embeddings', n_layers=tc, d_hidden=16, activation=jax.nn.gelu)
        model_obs = csmdp.Observation(d_state=sd, d_obs=od, std=os, n_layers=oc, d_hidden=16, activation=jax.nn.gelu)
        model_rew = csmdp.Reward(d_state=sd, std=rs, sparse=rspr, sparse_prob=rsprp, n_layers=rc, d_hidden=16, activation=jax.nn.gelu)
        model_done = csmdp.Done(d_state=sd, std=ds, sparse_prob=dsprp, n_layers=dc, d_hidden=16, activation=jax.nn.gelu)
        env = smdp.SyntheticMDP(model_init, model_trans, model_obs, model_rew, model_done)
        env = TimeLimit(env, n_steps_max=tl)
    elif env_cfg['name'] == 'hsmdp':
        sd1 = [2, 4, 6, 8, 16][int(env_cfg['sd'])]
        sd2 = [2, 4, 8, 16, 32][int(env_cfg['sd'])]
        ad = [1, 2, 3, 5, 8][int(env_cfg['ad'])]
        od = [2, 4, 8, 16, 32][int(env_cfg['od'])]
        isv1 = [.01, .1, .3, 1, 100][int(env_cfg['isv'])]
        isv2 = [0, 0.1, 0.3, 1.0, 3.0][int(env_cfg['isv'])]
        tc = [0, 1, 2, 4, 8][int(env_cfg['tc'])]
        tloc = [0.01, 0.03, 0.1, 0.3, 1.0][int(env_cfg['tloc'])]
        ts1 = [.01, .1, .3, 1, 100][int(env_cfg['ts'])]
        ts2 = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['ts'])]
        oc = [0, 1, 2, 4, 8][int(env_cfg['oc'])]
        os = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['os'])]
        rc = [0, 1, 2, 4, 8][int(env_cfg['rc'])]
        rs = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['rs'])]
        rspr = [False, True][int(env_cfg['rspr'])]
        rsprp = [0.01, 0.03, 0.1, 0.3, 0.9][int(env_cfg['rsprp'])]
        dc = [0, 1, 2, 4, 8][int(env_cfg['dc'])]
        ds = [0, 0.03, 0.1, 0.3, 1.0][int(env_cfg['ds'])]
        dsprp = [0.0, 0.01, 0.03, 0.1, 0.3][int(env_cfg['dsprp'])]
        tl = [1, 4, 16, 64, 128][int(env_cfg['tl'])]
        model_init = hsmdp.Init(m=sd1, n=sd1, d_state=sd2, temperature=isv1, std=isv2, constraint="clip", n_embeds=None)
        model_trans = hsmdp.Transition(m=sd1, n=sd1, d_state=sd2, n_acts=ad, temperature=ts1, std=ts2, locality=tloc, constraint='clip', n_layers=tc, d_hidden=16, activation=jax.nn.gelu)
        model_obs = hsmdp.Observation(m=sd1, n=sd1, d_state=sd2, d_obs=od, std=os, n_layers=oc, d_hidden=16, activation=jax.nn.gelu)
        model_rew = hsmdp.Reward(m=sd1, n=sd1, d_state=sd2, std=rs, sparse=rspr, sparse_prob=rsprp, n_layers=rc, d_hidden=16, activation=jax.nn.gelu)
        model_done = hsmdp.Done(m=sd1, n=sd1, d_state=sd2, std=ds, sparse_prob=dsprp, n_layers=dc, d_hidden=16, activation=jax.nn.gelu)
        env = smdp.SyntheticMDP(model_init, model_trans, model_obs, model_rew, model_done)
        env = TimeLimit(env, n_steps_max=tl)
    else:
        raise NotImplementedError
    if log_wrapper:
        env = LogWrapper(env)
    return env


def temp():
    if env_cfg['name'] == 'csmdp':
        # d_state, d_obs, n_acts = int(env_cfg['d_state']), int(env_cfg['d_obs']), int(env_cfg['n_acts'])
        i_d = [2, 4, 8, 16, 32][int(env_cfg['i_d'])]
        i_s = [0, 1e-1, 3e-1, 1e0, 3e0][int(env_cfg['i_s'])]
        model_init = csmdp.Init(d_state=i_d, std=i_s)

        t_a = [1, 2, 3, 4, 5][int(env_cfg['t_a'])]
        t_c = [0, 1, 2, 4, 8][int(env_cfg['t_c'])]
        t_l = [1e-2, 3e-2, 1e-1, 3e-1, 1e0][int(env_cfg['t_l'])]
        t_s = [0, 3e-2, 1e-1, 3e-1, 1e0][int(env_cfg['t_s'])]
        model_trans = csmdp.DeltaTransition(d_state=i_d, n_acts=t_a,
                                            n_layers=t_c, d_hidden=16, activation=nn.relu,
                                            locality=t_l, std=t_s)

        o_d = [2, 4, 8, 16, 32][int(env_cfg['o_d'])]
        o_c = [0, 1, 2, 4, 8][int(env_cfg['o_c'])]
        model_obs = csmdp.Observation(d_state=i_d, d_obs=o_d, n_layers=o_c, d_hidden=16, activation=nn.relu, std=0.)

        r_c = [0, 1, 2, 4, 8][int(env_cfg['r_c'])]
        model_rew = csmdp.DenseReward(d_state=i_d, n_layers=r_c, d_hidden=16, activation=nn.relu, std=0.)

        model_done = smdp.NeverDone()
        env = smdp.SyntheticMDP(model_init, model_trans, model_obs, model_rew, model_done)
    elif env_cfg['name'] == 'dsmdp':
        i_d = [8, 16, 32, 64, 128][int(env_cfg['i_d'])]
        i_s = [1000., 4, 2, 1, 0][int(env_cfg['i_s'])]
        model_init = dsmdp.Init(i_d, std=i_s)

        t_a = [1, 2, 3, 4, 5][int(env_cfg['t_a'])]
        t_s = [1000., 5, 4, 3, 2][int(env_cfg['i_s'])]
        model_trans = dsmdp.Transition(i_d, n_acts=t_a, std=t_s)

        o_d = [2, 4, 8, 16, 32][int(env_cfg['o_d'])]
        model_obs = dsmdp.Observation(i_d, d_obs=o_d, std=0.)

        model_rew = dsmdp.DenseReward(i_d, std=0.)

        model_done = smdp.NeverDone()

        env = smdp.SyntheticMDP(model_init, model_trans, model_obs, model_rew, model_done)
    else:
        raise NotImplementedError
    return env

