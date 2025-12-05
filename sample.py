import numpy as np
import matplotlib.pyplot as plt

import argparse
import os
import glob
import torch

from train import DiffusionModel
from faster_diffusion_dataset import full_diffusion_dataset

import time as timex

from scipy.special import expit,logit

def rnd_plotter(temp_init, seed, ifEMA, hps, im_id, idx, threshold=0.5):

    if np.random.rand(1) > threshold:
        if hps['data'] == 'mnist' or hps['data'] == 'cifar_gray':
            min_lim = None if hps['data'] == 'cifar_gray' else 0
            fig = plt.figure()
            plt.imshow(temp_init[0,], cmap='gray', vmin=min_lim, vmax=None)
            plt.colorbar()
            plt.title(f'{time:.4f}')
            plt.show()
            loss = hps['loss']
            schedule = hps['schedule']
            fig.tight_layout()
            if savefig:
                fig.savefig(f'{figFolder}/ver-{version}_loss-{loss}_sch-{schedule}_it-{trainingIteration}K_im-{im_id}_CFL-{max_moving_fraction}_tc-{time:.2f}_seed-{seed}_{idx}.png', dpi=200)
            plt.close()

        elif hps['data'] == 'cifar10':
            fig = plt.figure()
            plt.imshow(temp_init.transpose((1, 2, 0)))
            loss = hps['loss']
            schedule = hps['schedule']
            plt.title(f'ver {version}, loss: {loss}, schedule: {schedule}, im: {im_id}\n{trainingIteration}K iterations, CFL: {max_moving_fraction}, tc={time:.2f}, seed={seed}, ema={ifEMA}')
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plt.show()
            fig.tight_layout()
            if savefig:
                fig.savefig(f'{figFolder}/ver-{version}_loss-{loss}_sch-{schedule}_it-{trainingIteration}K_im-{im_id}_CFL-{max_moving_fraction}_tc-{time:.2f}_seed-{seed}_ema-{ifEMA}_{idx}.png', dpi=200)
            plt.close()

def reverse_sample_directions_torch(total_moving_particles, im_size_x, im_size_y, im_size_c, prob):
    bs = total_moving_particles.size(0)
    moving_particles = torch.zeros((bs, 4, im_size_c, im_size_x, im_size_y), dtype=torch.int64, device=device)
    for i in range(bs):
        moving_particles[i] = reverse_sample_directions_torch_inst(total_moving_particles[i], im_size_x, im_size_y, im_size_c, prob[i])
        #if torch.sum(moving_particles[i]) != torch.sum(total_moving_particles[i]):
        #    print("Errr", "in=", torch.sum(moving_particles[i][None,...]), "out=", torch.sum(total_moving_particles[i]))
    return moving_particles

def reverse_sample_directions_torch_inst(total_moving_particles, im_size_x, im_size_y, im_size_c, prob):
    device = total_moving_particles.device
    moving_particles = torch.zeros((4, im_size_c, im_size_x, im_size_y), dtype=torch.int64, device=device)

    indices = torch.nonzero(total_moving_particles[0], as_tuple=False)
    if indices.numel() == 0:
        return moving_particles

    c_idx, i_idx, j_idx = indices[:, 0], indices[:, 1], indices[:, 2]
    N = indices.shape[0]
    total_counts = total_moving_particles[0, c_idx, i_idx, j_idx].to(torch.int32)

    max_count = total_counts.max().item()

    # Sample uniform random values: shape (N, max_count)
    rand = torch.rand((N, max_count), device=device)

    # Get cumulative distribution of probabilities: shape (N, 4)
    probs = prob[:, c_idx, i_idx, j_idx].T  # (N, 4)
    cum_probs = probs.cumsum(dim=1)

    # Expand cum_probs to shape (N, max_count, 4) and rand to (N, max_count, 1)
    cum_probs_exp = cum_probs[:, None, :]  # (N, 1, 4)
    rand_exp = rand[:, :, None]  # (N, max_count, 1)

    # Compare and find sampled direction index per random sample
    sampled_dir = (rand_exp < cum_probs_exp).float().argmax(dim=2)  # (N, max_count)

    # Only keep valid samples (i.e., fewer than total_counts)
    mask = torch.arange(max_count, device=device)[None, :] < total_counts[:, None]
    valid_samples = sampled_dir[mask]  # flattened samples

    # Rebuild matching index array to count into output
    full_idx = indices.repeat_interleave(total_counts, dim=0)
    c_idx_r, i_idx_r, j_idx_r = full_idx[:, 0], full_idx[:, 1], full_idx[:, 2]

    # Count occurrences
    for d in range(4):
        dir_mask = valid_samples == d
        c = c_idx_r[dir_mask]
        i = i_idx_r[dir_mask]
        j = j_idx_r[dir_mask]
        moving_particles[d].index_put_((c, i, j), torch.ones_like(c, dtype=torch.int64), accumulate=True)

    return moving_particles


def tc2tp(tc, hps):

    if hps['schedule'] ==  'quadratic':
        tp = tc**2
    elif hps['schedule'] == 'power3':
        tp = tc**3
    elif hps['schedule'] == 'power4':
        tp = tc**4
    elif hps['schedule'] == 'power5':
        tp = tc**5
    elif hps['schedule'] == 'power6':
        tp = tc**6
    elif hps['schedule'] == 'power7':
        tp = tc**7
    elif hps['schedule'] == 'blackout':
        tEnd1 = 7.5
        tEnd2 = 2.5
        tp =  - np.log(expit( logit(1-np.exp(-tEnd1)) + (hps['t_size']*tc-1)/(hps['t_size']-2)*(logit(np.exp(-tEnd1))-logit(1-np.exp(-tEnd2)) )  ))/tEnd2
    else:
        raise NotImplementedError

    return tp


def tp2tc(tp, hps):

    if hps['schedule'] ==  'quadratic':
        tc = tp**(1./2)
    elif hps['schedule'] == 'power3':
        tc = tp**(1./3)
    elif hps['schedule'] == 'power4':
        tc = tp**(1./4)
    elif hps['schedule'] == 'power5':
        tc = tp**(1./5)
    elif hps['schedule'] == 'power6':
        tc = tp**(1./6)
    elif hps['schedule'] == 'power7':
        tc = tp**(1./7)
    elif hps['schedule'] == 'blackout':
        tEnd1 = 7.5
        tEnd2 = 2.5
        tc = ((logit(np.exp(-tp*tEnd2)) + logit(np.exp(-tEnd1)))/(logit(np.exp(-tEnd1))+logit(np.exp(-tEnd2)))*(hps['t_size']-2) + 1)/hps['t_size']
    else:
        raise NotImplementedError

    return tc

parser = argparse.ArgumentParser(description='Test model checkpoints')
parser.add_argument('--version', type=int,default=4,help='Version number of the model checkpoints')
parser.add_argument('--tolerance', type=float, default=0.15, required=True, help='Tolerance for the CFL step')
parser.add_argument('--gpu', type=int, default=0, help='GPU index to use')
parser.add_argument('--savefig', type=int, default=1, help='(0/1) If to save the generated figures')
parser.add_argument('--seed', type=int, default=1234, help='Seed for the RNG')
parser.add_argument('--ifEMA', type=int, default=0, help='(0/1) If to use EMA for inference')
parser.add_argument('--im_id', type=int, default=17378, help='Initial image id for generating fully corrupted image')
parser.add_argument('--ens', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=8)

if __name__ == '__main__':
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    version = args.version
    savefig = args.savefig
    checkpoint_dir = f'lightning_logs/version_{version}/checkpoints/'
    checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
    checkpoint_paths = sorted(checkpoint_paths, key=lambda x: int(x.split('step=')[-1].split('.ckpt')[0]), reverse=True)
    trainingIteration=int(int(sorted(checkpoint_paths)[-1].split('step=')[1].split('.ckpt')[0])/1000)

    model = DiffusionModel.load_from_checkpoint(checkpoint_paths[0], map_location=device)
    model.to(device).eval()

    hps = model.hparams
    dataset = full_diffusion_dataset(hps['data'], hps['r'], hps['t_size'], hps['schedule'], hps['PBC'])

    bs = args.batch_size

    im_size_c, im_size_x, im_size_y = 3, 32, 32

    # random init
    #init = torch.randint(0, 255, (bs, im_size_c, im_size_x, im_size_y), device=device, dtype=torch.float32)

    for ttttt in range(5_000):

        # init from known dist, i.e dataset
        init = torch.zeros((bs, im_size_c, im_size_x, im_size_y), device=device)
        seeds = []
        im_nums = []
        for i in range(bs):
            seed = np.random.choice(10000)
            seeds.append(seed)
            im_num = np.random.choice(50000)
            im_nums.append(im_num)
            im, _ = dataset.__get_image__(im_num, -1, seed)
            init[i] = torch.from_numpy(im)
        time = (hps['t_size']-1)/hps['t_size']
        time_list = []
        max_moving_fraction = args.tolerance
    
        indices = np.arange(hps['in_ch']*4).reshape((hps['in_ch'],4)).T.reshape((hps['in_ch']*4,))
    
        outputTick=0.05
        targetTick = 1.0-outputTick
        tk1 = tc2tp(1./hps['t_size'], hps)
    
        ifLastSnapshot = False
        ifContinue = True
        ifEMA = 0
    
        figFolder = 'figs_batch'
        try:
            os.mkdir(figFolder)
        except:
            pass
    
        steps = 0
        start_time = timex.time()
        seed = 3000
    
        while ifContinue:
            try:
                temp_init = init.detach().clone()
                with torch.no_grad():
                    time_tensor = torch.tensor([time]*bs, dtype=torch.float32, device=device)
                    temp_init_tensor = temp_init
        
                    if ifEMA:
                        r_hat = model.ema(time_tensor, temp_init_tensor)
                    else:
                        r_hat = model(time_tensor, temp_init_tensor)
                        
                r_hat = r_hat[:, indices, :, :]
                r_hat = r_hat.view(bs, 4, im_size_c, im_size_x, im_size_y)   # change the permutation back to C,X,Y for consistency
        
                # Mitigating different scaling/parametrization for different losses
                if hps['loss']!='blackout':
                    r_hat = (temp_init_tensor>0).type(torch.float)*r_hat/temp_init_tensor.clamp(min=1e-6)
        
                # Boundary conditions
                if hps['PBC']!=1:
                    r_hat[:, 0, :, -1,  :] = 0 # right-hopping rate
                    r_hat[:, 1, :,  0,  :] = 0 # left-hopping rate
                    r_hat[:, 2, :,  :, -1] = 0 # up-hopping rate
                    r_hat[:, 3, :,  :,  0] = 0 # down-hopping rate
        
                r_sum = r_hat.sum(dim=1) # YT: we need to compute the total rate per pixel first...
                r_max = r_sum.max() # YT:... then taking thes statistics over C,X,Y
        
                adaptive_dt_physical = max_moving_fraction / r_max  # This makes more sense and ensures that the p for binomial sampling is bounded by 1
                # handle the last time step more carefully
                if ifLastSnapshot:            
                    ifContinue = False
                    dtk_physical = tc2tp(time, hps)
                                        
                if (tc2tp(time, hps)-adaptive_dt_physical) < tk1:
                    adaptive_dt_physical = torch.tensor(tc2tp(time, hps) - tk1).to(device)
                    ifLastSnapshot = True
        
                p_hat = r_hat*adaptive_dt_physical # changing to p_hat, as they are probability now, not transition rates
                total_p_hat = p_hat.sum(axis=1, keepdim=True)
                p_hat_norm = p_hat/total_p_hat
                # total_moving_particles = torch.distributions.binomial.Binomial(temp_init[:,None,:,:,:], total_p_hat).sample()
                total_moving_particles = torch.from_numpy(np.random.binomial(temp_init[:,None,:,:,:].cpu().int().numpy(), total_p_hat.cpu().numpy())).to(device)
                sampled_moves = reverse_sample_directions_torch(total_moving_particles, im_size_x, im_size_y, im_size_c, p_hat_norm)
        
                # debugging 
                # if wrong dimensions, particles will not be conserved
                # if torch.sum(sampled_moves) != torch.sum(total_moving_particles):
                #     raise Exception("number of particles changed ! in_particles = ", torch.sum(total_moving_particles), " out_particles = ", torch.sum(sampled_moves))
        
                if hps['PBC']!=1:
                    # YT: vectorize/matrix operation instead of looping over all the pixels & channels
                    temp_init[:,:, 1:,:] += sampled_moves[:,0,:,:-1,:] # R, influx
                    temp_init[:,:,:-1,:] -= sampled_moves[:,0,:,:-1,:] # R, efflux
                    temp_init[:,:,:-1,:] += sampled_moves[:,1,:, 1:,:] # L, influx
                    temp_init[:,:, 1:,:] -= sampled_moves[:,1,:, 1:,:] # L, efflux
                    temp_init[:,:,:, 1:] += sampled_moves[:,2,:,:,:-1] # U, influx
                    temp_init[:,:,:,:-1] -= sampled_moves[:,2,:,:,:-1] # U, efflux
                    temp_init[:,:,:,:-1] += sampled_moves[:,3,:,:, 1:] # D, influx
                    temp_init[:,:,:, 1:] -= sampled_moves[:,3,:,:, 1:] # D, efflux   
                else:
                    temp_init[:,:,:,:] -= sampled_moves[:,0,:,:,:] # R, efflux
                    temp_init[:,:,:,:] -= sampled_moves[:,1,:,:,:] # L, efflux
                    temp_init[:,:,:,:] -= sampled_moves[:,2,:,:,:] # U, efflux
                    temp_init[:,:,:,:] -= sampled_moves[:,3,:,:,:] # D, efflux
                    temp_init[:,:,:,:] += torch.roll(sampled_moves[:,0,:,:,:], 1, dims=2)# R, influx
                    temp_init[:,:,:,:] += torch.roll(sampled_moves[:,1,:,:,:], -1, dims=2) # L, influx
                    temp_init[:,:,:,:] += torch.roll(sampled_moves[:,2,:,:,:], 1, dims=3) # U, influx
                    temp_init[:,:,:,:] += torch.roll(sampled_moves[:,3,:,:,:], -1, dims=3) # D, influx
                            
                init = temp_init.detach().clone()
                
                # time stepping
                time_physical =  tc2tp(time, hps) - adaptive_dt_physical.cpu().numpy()
        
                if ifContinue:
                    time = tp2tc(time_physical, hps)
                else:
                    time = 0
                            
                time_list.append([time, time_physical, adaptive_dt_physical.detach().cpu().numpy()])
        
                rtime = timex.time()
                if time <= targetTick:
                    print(f'Steps {steps}, took {rtime - start_time}s')
                    print(f'Time {time:.5f} Success')
                    targetTick -= outputTick
            except Exception as e:
                    print("An error occurred:", e)
                    #rnd_plotter(temp_init, hps, threshold=0.0)
                    break
            steps += 1
    
        end_time = timex.time()
        print(f"Finished generating {bs} images in {end_time-start_time}s")
        for i in range(bs):
            rnd_plotter(temp_init[i].int().detach().cpu().numpy(), seeds[i], ifEMA, hps, im_nums[i], ttttt*bs + i, threshold=0.0)