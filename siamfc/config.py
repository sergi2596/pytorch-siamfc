class Config:

    # dataset related
    exemplar_size = 127                    # exemplar size
    instance_size = 255                    # instance size
    context_amount = 0.5                   # context amount

    # training related
    num_per_epoch = 20000                  # num of samples per epoch
    train_ratio = 0.8                      # training ratio of VID dataset
    frame_range = 100                      # frame range of choosing the instance
    train_batch_size = 8                   # training batch size
    valid_batch_size = 8                   # validation batch size
    train_num_workers = 8                  # number of workers of train dataloader
    valid_num_workers = 8                  # number of workers of validation dataloader
    lr = 1e-3                              # learning rate of SGD
    momentum = 0.0                         # momentum of SGD
    weight_decay = 5e-4                    # weight decay of optimizator
    step_size = 25                         # step size of LR_Schedular
    gamma = 0.1                            # decay rate of LR_Schedular
    start_epoch = 0                        # start from epoch
    epoch = 50                             # total epoch
    seed = 1234                            # seed to sample training videos
    # experiment_folder = 'experiments'      # Experiment dirs
    test_folder = 'tracker_test'           # Tracker test dirs
    radius = 16                            # radius of positive label
    response_scale = 1e-5                  # normalize of response
    max_translate = 3                      # max translation of random shift

    # tracking related
    scale_step = 1.0375                    # scale step of instance image
    num_scale = 1                          # number of scales
    scale_lr = 0.59                        # scale learning rate
    response_up_stride = 16                # response upsample stride
    response_sz = 17                       # response size
    train_response_sz = 15                 # train response size
    window_influence = 0.176               # window influence
    scale_penalty = 0.9745                 # scale penalty
    total_stride = 8                       # total stride of backbone
    sample_type = 'uniform'
    gray_ratio = 0.25
    blur_ratio = 0.15
    model_path = 'models/siamfc_50.pth'


config = Config()
    
    # p.numScale = 5;
    # p.scaleStep = 1.0255;
    # p.scalePenalty = 0.962;  % penalizes the change of scale
    # p.scaleLR = 0.34;
    # p.responseUp = 16; % response upsampling factor (purpose is to account for stride, but they dont have to be equal)
    # p.windowing = 'cosine';
    # p.wInfluence = 0.168;
    # p.net = '2016-08-17.net.mat';
