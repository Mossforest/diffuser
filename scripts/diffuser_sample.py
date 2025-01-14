import diffuser.utils as utils

class Args:
    loadpath = 'logs/pretrained/hopper-medium-expert-v2/diffusion/defaults_H32_T20'
    diffusion_epoch = 'latest'
    n_samples = 4
    device = 'cuda:0'


if __name__ == '__main__':
        
    args = Args()

    diffusion_experiment = utils.load_diffusion(
        args.loadpath, epoch=args.diffusion_epoch)

    dataset = diffusion_experiment.dataset
    renderer = diffusion_experiment.renderer
    model = diffusion_experiment.trainer.ema_model

    env = dataset.env
    obs = env.reset()

    observations = utils.colab.run_diffusion(
        model, dataset, obs, args.n_samples, args.device)
    print(observations.shape)
    
    sample = observations[-1]
    breakpoint()
    utils.colab.show_sample(renderer, sample)