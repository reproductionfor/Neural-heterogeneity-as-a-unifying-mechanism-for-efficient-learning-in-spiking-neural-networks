import torch
import torch.nn as nn
import numpy as np
import math
from scipy.stats import cauchy, norm
from utils import set_seed
from SuSpike import SuSpike
spike_fn = SuSpike.apply

def lorentzian(n: int, eta: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = cauchy.rvs(loc=eta, scale=delta)
        while s <= lb or s >= ub:
            s = cauchy.rvs(loc=eta, scale=delta)
        samples[i] = s
    return samples
def gaussian(n: int, eta: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = norm.rvs(loc=eta, scale=delta)
        while s <= lb or s >= ub:
            s = norm.rvs(loc=eta, scale=delta)
        samples[i] = s
    return samples

def dist_fn(dist):
    return {
        'gamma': lambda mean, k, size: np.random.gamma(k, scale=mean/k, size=size),
        'normal': lambda mean, k, size: np.random.normal(loc=mean, scale=mean/np.sqrt(k), size=size), #change standard deviation to match gamma
        'uniform': lambda _, maximum, size: np.random.uniform(low=0, high=maximum, size=size),
    }[dist.lower()]


class SpikingLayer(nn.Module):
    # Implements LIF synaptic filtering + membrane filtering
    def __init__(self, input_size, output_size, prms):
        super(SpikingLayer, self).__init__()
        self.device = prms['device']
        self.dtype = prms['dtype']
        self.tref = prms['tref']
        self.dist_prms = prms['dist_prms']
        set_seed(prms['seed'])

        # Create variables
        self.w = nn.Parameter(
            torch.empty((input_size, output_size), device=self.device, dtype=self.dtype, requires_grad=True))
        self.alpha = nn.Parameter(torch.empty((1, output_size), device=self.device, dtype=self.dtype), requires_grad=bool(prms['train_ab']))
        self.beta = nn.Parameter(torch.empty((1, output_size), device=self.device, dtype=self.dtype), requires_grad=bool(prms['train_ab']))
        self.th = nn.Parameter(torch.empty((1, output_size), device=self.device, dtype=self.dtype), requires_grad=bool(prms['train_th']))

        # Init variables
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.w, -bound, bound)

        if prms['het_ab']:
            distribution = dist_fn(prms['dist'])
            dist_alpha = np.exp(-prms['time_step'] / distribution(prms['tau_syn'], self.dist_prms, (1, output_size)))
            dist_beta = np.exp(-prms['time_step'] / distribution(prms['tau_mem'], self.dist_prms, (1, output_size)))
            self.alpha.data.copy_(torch.from_numpy(dist_alpha))
            self.beta.data.copy_(torch.from_numpy(dist_beta))
        else:
            nn.init.constant_(self.alpha, prms['alpha'])
            nn.init.constant_(self.beta, prms['beta'])
        if prms['het_th']:
            nn.init.uniform_(self.th, a=0.5, b=1.5)
        else:
            nn.init.constant_(self.th, 1.)

        self.nb_steps = prms['nb_steps']
        self.output_size = output_size


    def forward(self, inputs):
        h = torch.einsum("abc,cd->abd", (inputs[-1], self.w))

        batch_size = h.shape[0]

        syn = torch.zeros((batch_size, self.output_size), device=self.device, dtype=self.dtype)
        mem = torch.zeros((batch_size, self.output_size), device=self.device, dtype=self.dtype)
        ref = torch.zeros((batch_size, self.output_size), device=self.device, dtype=self.dtype)

        syn_rec = [syn]
        mem_rec = [mem]
        spk_rec = [mem]

        # Compute hidden layer activity
        for t in range(self.nb_steps - 1):
            mthr = mem - self.th
            out = spike_fn(mthr)
            rst = torch.zeros_like(mem)
            c = (mthr > 0)
            rst[c] = torch.ones_like(mem)[c]
            ref[c] = self.tref * torch.ones_like(mem)[c]

            new_syn = self.alpha * syn + h[:, t]
            # new_mem = (self.beta * mem + self.rest + syn - rst*self.th.detach()) * (ref[:] < 1.).type(self.dtype)
            new_mem = (self.beta * mem + (1 - self.beta.detach())*syn - rst*self.th.detach()) * (ref[:] < 1.).type(self.dtype)

            ref[ref[:] > 0.] -= 1.

            mem = new_mem
            syn = new_syn

            syn_rec.append(syn)
            mem_rec.append(mem)
            spk_rec.append(out)

        syn_rec = torch.stack(syn_rec, dim=1)
        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)

        return (syn_rec, mem_rec, spk_rec)


class MembraneLayer(nn.Module):
    # Implements LIF synaptic filtering + membrane filtering
    def __init__(self, input_size, output_size, prms):
        super(MembraneLayer, self).__init__()
        self.device = prms['device']
        self.dtype = prms['dtype']
        self.dist_prms = prms['dist_prms']
        set_seed(prms['seed'])

        # Variables in each layer
        self.w = nn.Parameter(torch.empty((input_size, output_size), device=self.device, dtype=self.dtype, requires_grad=True))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.w, -bound, bound)

        # self.alpha = nn.Parameter(torch.empty((1, output_size), device=self.device, dtype=self.dtype), requires_grad=bool(prms['train_ab']))
        # self.beta = nn.Parameter(torch.empty((1, output_size), device=self.device, dtype=self.dtype), requires_grad=bool(prms['train_ab']))
        self.alpha = nn.Parameter(torch.empty((1, output_size), device=self.device, dtype=self.dtype), requires_grad=False)
        self.beta = nn.Parameter(torch.empty((1, output_size), device=self.device, dtype=self.dtype), requires_grad=False)

        if prms['het_ab']:
            distribution = dist_fn(prms['dist'])
            dist_alpha = np.exp(-prms['time_step'] / distribution(prms['tau_syn'], self.dist_prms, (1, output_size)))
            dist_beta = np.exp(-prms['time_step'] / distribution(prms['tau_mem'], self.dist_prms, (1, output_size)))
            self.alpha.data.copy_(torch.from_numpy(dist_alpha))
            self.beta.data.copy_(torch.from_numpy(dist_beta))
        else:
            if not prms['train_ab']:
                self.alpha = nn.Parameter(torch.empty((1), device=self.device, dtype=self.dtype), requires_grad=False)
                self.beta = nn.Parameter(torch.empty((1), device=self.device, dtype=self.dtype), requires_grad=False)
            nn.init.constant_(self.alpha, prms['alpha'])
            nn.init.constant_(self.beta, prms['beta'])

        self.nb_steps = prms['nb_steps']
        self.output_size = output_size

    def forward(self, inputs):
        h = torch.einsum("abc,cd->abd", (inputs[-1], self.w))

        batch_size = h.shape[0]

        syn = torch.zeros((batch_size, self.output_size), device=self.device, dtype=self.dtype)

        mem = torch.zeros((batch_size, self.output_size), device=self.device, dtype=self.dtype)


        syn_rec = [syn]
        mem_rec = [mem]

        # Compute hidden layer activity
        for t in range(self.nb_steps - 1):
            new_syn = self.alpha * syn + h[:, t]
            # new_mem = self.beta * mem + syn
            new_mem = self.beta * mem + (1 - self.beta.detach())*syn

            mem = new_mem
            syn = new_syn

            syn_rec.append(syn)
            mem_rec.append(mem)

        syn_rec = torch.stack(syn_rec, dim=1)
        mem_rec = torch.stack(mem_rec, dim=1)
        #breakpoint()
        return (syn_rec, mem_rec)


class SynapticLayer(nn.Module):
    # Implements LIF synaptic filtering + membrane filtering
    def __init__(self, input_size, output_size, prms):
        super(SynapticLayer, self).__init__()
        self.device = prms['device']
        self.dtype = prms['dtype']
        self.dist_prms = prms['dist_prms']
        set_seed(prms['seed'])

        # Variables in each layer
        self.w = nn.Parameter(torch.empty((input_size, output_size), device=self.device, dtype=self.dtype, requires_grad=True))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.w, -bound, bound)

        self.alpha = nn.Parameter(torch.empty((1, output_size), device=self.device, dtype=self.dtype), requires_grad=bool(prms['train_ab']))

        if prms['het_ab']:
            distribution = dist_fn(prms['dist'])
            dist_alpha = np.exp(-prms['time_step'] / distribution(prms['tau_syn'], self.dist_prms, (1, output_size)))
            self.alpha.data.copy_(torch.from_numpy(dist_alpha))
        else:
            nn.init.constant_(self.alpha, prms['alpha'])

        self.nb_steps = prms['nb_steps']
        self.output_size = output_size

    def forward(self, inputs):
        h = torch.einsum("abc,cd->abd", (inputs[-1], self.w))
        batch_size = h.shape[0]
        syn = torch.zeros((batch_size, self.output_size), device=self.device, dtype=self.dtype)

        syn_rec = [syn]

        # Compute hidden layer activity
        for t in range(self.nb_steps - 1):
            # new_syn = self.alpha * syn + h[:, t]
            new_syn = self.alpha * (1 - self.beta.detach())*syn + h[:, t]

            syn = new_syn

            syn_rec.append(syn)

        syn_rec = torch.stack(syn_rec, dim=1)

        return (syn_rec, )


class PerceptronLayer(nn.Module):
    # Implements LIF perceptron layer (just einsum of spiking layer)
    def __init__(self, input_size, output_size, prms):
        super(PerceptronLayer, self).__init__()
        self.device = prms['device']
        self.dtype = prms['dtype']
        set_seed(prms['seed'])

        # Variables in each layer
        self.w = nn.Parameter(torch.empty((input_size, output_size), device=self.device, dtype=self.dtype, requires_grad=True))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.w, -bound, bound)

        self.nb_steps = prms['nb_steps']
        self.output_size = output_size

    def forward(self, inputs):
        h = torch.einsum("abc,cd->abd", (inputs[1], self.w))
        return h


class RecurrentSpikingLayer(nn.Module):
    # Implements LIF synaptic filtering + membrane filtering
    def __init__(self, input_size, output_size, prms):
        super(RecurrentSpikingLayer, self).__init__()
        self.device = prms['device']
        self.dtype = prms['dtype']
        self.dist_prms = prms['dist_prms']
        set_seed(prms['seed'])

        # Create variables
        self.w = nn.Parameter(
            torch.empty((input_size, output_size), device=self.device, dtype=self.dtype, requires_grad=True))
        self.v = nn.Parameter(
            torch.empty((output_size, output_size), device=self.device, dtype=self.dtype, requires_grad=True))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.w, -bound, bound)

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.v)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.v, -bound, bound)
        self.a = nn.Parameter(torch.full((1, output_size), prms['a'], device=self.device, dtype=self.dtype), requires_grad=False)
        #self.theta = nn.Parameter(torch.full((1, output_size), prms['theta'], device=self.device, dtype=self.dtype), requires_grad=False)

        theta_1 = lorentzian(output_size, eta=prms['theta'], delta=prms['delta_theta'], lb=-0.1, ub=1.1)
        self.theta = nn.Parameter(torch.tensor(theta_1, device=self.device, dtype=self.dtype).unsqueeze(0),
                              requires_grad=bool(prms['train_theta']))
        #breakpoint()

        self.b = nn.Parameter(torch.full((1, output_size), prms['b'], device=self.device, dtype=self.dtype), requires_grad=False)
        #self.g = nn.Parameter(torch.full((1, output_size), prms['g'], device=self.device, dtype=self.dtype), requires_grad=False)

        g_g = lorentzian(output_size, eta=prms['g'], delta=prms['delta_g'], lb=0, ub=2 * prms['g'])
        #g_g = gaussian(output_size, eta=prms['g'], delta=prms['delta_g'], lb=0, ub=2 * prms['g'])
        self.g = nn.Parameter(torch.tensor(g_g, device=self.device, dtype=self.dtype).unsqueeze(0),
                                requires_grad=bool(prms['train_g']))

        self.er = nn.Parameter(torch.full((1, output_size), prms['er'], device=self.device, dtype=self.dtype), requires_grad=False)
        self.alpha = nn.Parameter(torch.empty((1, output_size), device=self.device, dtype=self.dtype), requires_grad=bool(prms['train_ab']))
        self.beta = nn.Parameter(torch.empty((1, output_size), device=self.device, dtype=self.dtype), requires_grad=bool(prms['train_ab']))
        self.w_jump = nn.Parameter(torch.full((1, output_size), prms['w_jump'], device=self.device, dtype=self.dtype), requires_grad=False)
        self.dt = prms['time_step']
        self.tau_mem = prms['tau_mem']
        self.alpha1 = prms['alpha1']
        if prms['het_ab']:
            distribution = dist_fn(prms['dist'])
            dist_alpha = np.exp(-prms['time_step'] / distribution(prms['tau_syn'], self.dist_prms, (1, output_size)))
            dist_beta = np.exp(-prms['time_step'] / distribution(prms['tau_mem'], self.dist_prms, (1, output_size)))
            self.alpha.data.copy_(torch.from_numpy(dist_alpha))
            self.beta.data.copy_(torch.from_numpy(dist_beta))
        else:
            if not prms['train_ab']:
                self.alpha = nn.Parameter(torch.empty((1), device=self.device, dtype=self.dtype), requires_grad=bool(prms['train_hom_ab']))
                self.beta = nn.Parameter(torch.empty((1), device=self.device, dtype=self.dtype), requires_grad=bool(prms['train_hom_ab']))
            nn.init.constant_(self.alpha, prms['alpha'])
            nn.init.constant_(self.beta, prms['beta'])
        # self.eta = nn.Parameter(torch.empty((1, output_size), device=self.device, dtype=self.dtype), requires_grad=False)
        # eta_h = lorentzian(output_size, eta=prms['eta'], delta=prms['delta'], lb=-prms['eta'], ub=3 * prms['eta'])
        # self.eta.data.copy_(torch.from_numpy(eta_h))
        #breakpoint()
        #eta_h = lorentzian(output_size, eta=prms['eta'], delta=prms['delta'], lb=0, ub=2 * prms['eta'])
        eta_h = gaussian(output_size, eta=prms['eta'], delta=prms['delta'], lb=0, ub=2 * prms['eta'])
        self.eta = nn.Parameter(torch.tensor(eta_h, device=self.device, dtype=self.dtype).unsqueeze(0),
                                requires_grad=bool(prms['train_eta']))
        #breakpoint()
        self.th = nn.Parameter(torch.empty((1, output_size), device=self.device, dtype=self.dtype), requires_grad=bool(prms['train_th']))

        if prms['het_th']:
            nn.init.uniform_(self.th, a=0.5, b=1.5)
        else:
            nn.init.constant_(self.th, 1.)

        self.reset = nn.Parameter(torch.empty((1, output_size), device=self.device, dtype=self.dtype), requires_grad=bool(prms['train_reset']))
        if prms['het_reset']:
            nn.init.uniform_(self.reset, a=-0.5, b=0.5)
        else:
            nn.init.constant_(self.reset, 0.)

        self.rest = nn.Parameter(torch.empty((1, output_size), device=self.device, dtype=self.dtype), requires_grad=bool(prms['train_rest']))
        if prms['het_rest']:
            nn.init.uniform_(self.rest, a=-0.5, b=0.5)
        else:
            nn.init.constant_(self.rest, 0.)

        self.nb_steps = prms['nb_steps']
        self.output_size = output_size

    def forward(self, inputs):
        h = torch.einsum("abc,cd->abd", (inputs[-1], self.w))
        #breakpoint()
        batch_size = h.shape[0]
        #breakpoint()
        syn = torch.zeros((batch_size, self.output_size), device=self.device, dtype=self.dtype)
        mem = torch.zeros((batch_size, self.output_size), device=self.device, dtype=self.dtype)
        recover = torch.zeros((batch_size, self.output_size), device=self.device, dtype=self.dtype)
        syn_rec = [syn]
        mem_rec = [mem]
        recover_rec = [recover]
        spk_rec = [mem]

        # Compute hidden layer activity
        for t in range(self.nb_steps - 1):
            mthr = mem - self.th

            out = spike_fn(mthr)
            rst = torch.zeros_like(mem)
            c = (mthr > 0)
            rst[c] = torch.ones_like(mem)[c]

            new_syn = self.alpha * syn + h[:, t] + torch.mm(out, self.v)
            # new_mem = self.beta * mem + syn - rst
            #new_mem = self.beta * (mem - self.rest) + self.rest + (1 - self.beta.detach())*syn - rst*(self.th.detach()-self.reset)
            new_mem = mem+self.dt*(mem*(mem-self.alpha1)+self.eta-recover+self.g*syn*(self.er-mem))/self.tau_mem
            new_recover = recover+self.dt*(self.a*(self.b*mem-recover))
            new_recover = new_recover + rst*(-self.dt*(self.a*(self.b*mem-recover))+self.w_jump)
            # new_mem = new_mem+rst*(-new_mem+self.theta*(mem-self.th))
            new_mem = new_mem + rst * (-new_mem + self.theta * (mem-self.th))
            #breakpoint()
            mem = new_mem
            syn = new_syn
            recover = new_recover
            syn_rec.append(syn)
            mem_rec.append(mem)
            recover_rec.append(recover)
            spk_rec.append(out)

        syn_rec = torch.stack(syn_rec, dim=1)
        mem_rec = torch.stack(mem_rec, dim=1)
        recover_rec = torch.stack(recover_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)

        return (syn_rec, mem_rec,recover_rec, spk_rec)
