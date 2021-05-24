# Diabetic Retinopathy Detection 
This Repo contains my scripts for the [
SETI Breakthrough Listen - E.T. Signal Search](https://www.kaggle.com/c/seti-breakthrough-listen) Competition.

## Papers
## Features
- &#x2611; Balanced Sampler 

- &#x2611; Mixed Precision

- &#x2611; Gradient Accumulation  

- &#x2611; Optimum Learning Rate Finder [LR Finder Suggestion is terrible. I just observed the learning rate at which loss starts to diverge and set `learning_rate = learning rate at diverging loss/100`. No particular intention behind it.] 

- &#x2611; Out of Fold
## Resources

## Can be useful

## How to run
- Run `git clone https://github.com/tahsin314/SETI-Signal-Search`
- Run `conda env create -f environment.yml`
- Run `train.py`. Change parameters according to your preferences from the `dr_config.ini` file before training.
- `dr_config` parameters:
    ```
    n_fold = Total number of folds
    fold = fold that you want to keep as your validation set
    SEED = Seed value. This value will be use for random initialization
    batch_size 
    sz = Image Dimension
    learning_rate 
    patience = patience for Learning Rate Scheduler
    accum_step = Gradient Accumulation steps 
    num_class = If regression, it should be 1 otherwise 5
    gpu_ids = GPUs to use
    mixed_precision 
    pretrained_model = model name
    model_type = Normal, TripletAttention, CBAttention
    cam_layer_name = Class Activation Mapping Layer
    model_dir = Directory where models will be saved
    distributed_backend = None, dp or ddp. Currently ddp does not work
    mode = train or lr_finder
    load_model = 0 if False else 1
    imagenet_stats = mean and standard deviation of imagenet data
    n_epochs = number of epochs to train
    TTA = Test Time Augmentation
    oof = Out of Fold(1 if True)
    balanced_sampler 
    data_dir 
    image_path 
    test_image_path 
    ```

### One important thing about EfficientNet
EfficientNet's are designed to take in to account input image dimensions.

So if you want to squeeze every last droplet from your model make sure to use same image resolutions as described below:

```
Efficientnet-B0 : 224
Efficientnet-B1 : 240
Efficientnet-B2 : 260
Efficientnet-B3 : 300
Efficientnet-B4 : 380
Efficientnet-B5 : 456
Efficientnet-B6 : 528
Efficientnet-B7 : 600
```

## Issues
- Currently muli gpu training only works with `distributed_backend='dp'`. However, `Stochastic Weight Averaging` breaks it. 
- `distributed_backend='dp'` also fails to gather data from multiple GPUs and returns predictions and labels with shape `(batch_num, )` where it is supposed to return data with shape `(num_samples, )` (`num_samples ≈ num_batches * batch_size`). This is probably a *pytorch-lightning* bug. It is advised to comment out `distributed_backend='dp'` and training on single GPU at this moment. 

**Alternative:** Try running inference on test data on single GPU or set `batch_size=1` during test.
### Update [20/04/2021]
Added an ugly hack for gathering predictions and labels with `distributed_backend='dp'`. Now data can be gathered from multiple GPUs. Hope that *pytorch-Lightning* will fix their `gather` issue.

