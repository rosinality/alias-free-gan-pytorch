{
    generator: {
        __target: 'model.Generator',
        style_dim: 512,
        n_mlp: 2,
        kernel_size: 3,
        n_taps: 6,
        filter_parameters: {
            __target: 'model.filter_parameters',
            n_layer: 14,
            n_critical: 2,
            sr_max: $.training.size,
            cutoff_0: 2,
            cutoff_n: self.sr_max / 2,
            stopband_0: std.pow(2, 2.1),
            stopband_n: self.cutoff_n * std.pow(2, 0.3),
            channel_max: 512,
            channel_base: std.pow(2, 14)
        },
        margin: 10,
        lr_mlp: 0.01
    },

    discriminator: {
        __target: 'stylegan2.model.Discriminator',
        size: $.training.size,
        channel_multiplier: 2
    },

    training: {
        size: 256,
        iter: 800000,
        batch: 16,
        n_sample: 32,
        r1: 10,
        d_reg_every: 16,
        lr_g: 3e-3,
        lr_d: 2.5e-3,
        augment: false,
        augment_p: 0,
        ada_target: 0.6,
        ada_length: 500 * 1000,
        ada_every: 256,
        start_iter: 0
    }
}