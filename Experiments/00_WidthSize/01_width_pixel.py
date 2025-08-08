
# %%
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from typing import Any, Callable, Sequence, Union
import numpy as np

import tensorflow as tf
tf.config.set_visible_devices([], device_type='GPU')

import jax
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze, FrozenDict
from flax import linen as nn
from flax import struct
from flax.training import train_state
from flax.training import orbax_utils

import optax
import orbax.checkpoint

from clu import metrics
from ml_collections import ConfigDict

from einops import reduce, rearrange
import wandb
from fxlayers.layers import *
from fxlayers.layers import GaborLayerGamma_
from paramperceptnet.layers import pad_same_from_kernel_size
from functionalfourier.layers import *
from functionalfourier.layers import GaborReductionBlock
# from fxlayers.layers import GaborGammaFourier
from fxlayers.initializers import *
from JaxPlayground.utils.constraints import *
from JaxPlayground.utils.wandb import *

from functionalfourier.data import load_data

# %%
from tensorflow.keras.datasets import mnist, cifar10

# %%
config = {
    "DATASET": "mnist", # mnist / cifar10 / cats_vs_dogs
    "TEST_SPLIT": 0.2,
    "BATCH_SIZE": 64,
    "EPOCHS": 50,
    "LEARNING_RATE": 3e-3,
    "SEED": 42,

    "NORMALIZE_PROB": False,
    "NORMALIZE_ENERGY": True,
    "ZERO_MEAN": True,

    "USE_BIAS": False,
    "GAP": True,
    "SAME_REAL_IMAG": True,
    "N_SCALES": 4,
    "N_ORIENTATIONS": 8,

    "N_GABORS": 64,
    "REDUCTION": 0.5,
    "A_GABOR": True,
    # "A_GDNSPATIOFREQORIENT": True,

    "N_BLOCKS": 1,

    "USE_DNGAUSS": False,
    "USE_DNCONV": False,
    "DN_KERNEL_SIZE": 5,
    "GABOR_KERNEL_SIZE": 11,
}
config = ConfigDict(config)
config

# %%
wandb.init(project="FourierNN",
           name="GaborPixel",
           job_type="testing",
           config=dict(config),
           mode="disabled",
           )
config = wandb.config
print(config)

# %%
dst_train, dst_val = load_data(config.DATASET, test_split=config.TEST_SPLIT)
if config.DATASET == "cats_vs_dogs": N_CLASSES = 2
else: N_CLASSES = 10
config["N_CLASSES"] = N_CLASSES
len(dst_train), len(dst_val)

# %%
dst_train_rdy = dst_train.shuffle(buffer_size=100,
                              reshuffle_each_iteration=True,
                              seed=config.SEED)\
                         .batch(config.BATCH_SIZE, drop_remainder=True)
if dst_val is not None: 
     dst_val_rdy = dst_val.batch(config.BATCH_SIZE, drop_remainder=True)

# %%
class GaborReductionReLUBlock(nn.Module):
    n_scales: int
    n_orientations: int
    kernel_size: int
    reduction: float
    fs: float
    norm_prob: bool = False
    norm_energy: bool = True
    zero_mean: bool = True
    use_bias: bool = False
    train_A: bool = False

    @nn.compact
    def __call__(self,
                 inputs,
                 **kwargs,
                 ):
        ## Apply Gabor + Reduction block
        # outputs_r = GaborReductionBlock(features=self.features, reduction=self.reduction, fs=self.fs, norm_energy=self.norm_energy, same_real_imag=self.same_real_imag, train_A=self.train_A)(outputs_r, outputs_i)
        outputs = pad_same_from_kernel_size(
            inputs, kernel_size=self.kernel_size, mode="symmetric"
        )
        outputs = GaborLayerGamma_(
            n_scales=self.n_scales,
            n_orientations=self.n_orientations,
            kernel_size=self.kernel_size,
            fs=self.fs,
            xmean=self.kernel_size / 32 / 2,
            ymean=self.kernel_size / 32 / 2,
            strides=1,
            padding="VALID",
            normalize_prob=self.norm_prob,
            normalize_energy=self.norm_energy,
            zero_mean=self.zero_mean,
            use_bias=self.use_bias,
            train_A=self.train_A,
        )(outputs, return_freq=False, return_theta=False, **kwargs)

        ## Reduce the spatial dim
        outputs = nn.max_pool(outputs, window_shape=(int(1/self.reduction), int(1/self.reduction)))

        outputs = nn.relu(outputs)

        return outputs

# %%
class Model(nn.Module):
    config: Any

    @nn.compact
    def __call__(self,
                 inputs,
                 **kwargs,
                 ):
        outputs = inputs
        for i in range(self.config.N_BLOCKS):
            outputs = GaborReductionReLUBlock(n_scales=self.config.N_SCALES*(i+1), n_orientations=self.config.N_ORIENTATIONS, kernel_size=self.config.GABOR_KERNEL_SIZE,
                                              reduction=self.config.REDUCTION, fs=32*self.config.REDUCTION*(i+1)/self.config.REDUCTION, norm_energy=self.config.NORMALIZE_ENERGY,
                                              train_A=config.A_GABOR)(outputs, **kwargs)
    
        ## GAP & Dense for final prediction
        if self.config.GAP:
            outputs = reduce(outputs, "b h w c -> b c", "mean")
        else:
            outputs = rearrange(outputs, "b h w c -> b (h w c)")

        outputs = nn.Dense(features=self.config.N_CLASSES)(outputs)

        return outputs

# %%
model = Model(config)
variables = model.init(random.PRNGKey(42), jnp.ones((1,*next(iter(dst_train))[0].shape)))
# print(variables.keys())
state, params = flax.core.pop(variables, "params")
# print(f"State: {state}")
# print(f"Params: {params}")

# %%
@struct.dataclass
class Metrics(metrics.Collection):
    """Collection of metrics to be tracked during training."""
    loss: metrics.Average.from_output("loss")
    accuracy: metrics.Accuracy

# %%
class TrainState(train_state.TrainState):
    metrics: Metrics
    state: FrozenDict

# %%
def create_train_state(module, key, tx, input_shape):
    """Creates the initial `TrainState`."""
    variables = module.init(key, jnp.ones(input_shape))
    # state, params = variables.pop('params')
    state, params = flax.core.pop(variables, "params")
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        state=state,
        tx=tx,
        metrics=Metrics.empty()
    )

# %%
from functools import partial

# %%
@partial(jax.jit, static_argnums=2)
def train_step(state, batch, return_grads=False):
    """Train for a single step."""
    img, label = batch
    def loss_fn(params):
        ## Forward pass through the model
        pred, updated_state = state.apply_fn({"params": params, **state.state}, img, mutable=list(state.state.keys()), train=True)

        ## Calculate the distance
        loss = optax.softmax_cross_entropy_with_integer_labels(pred, label)
        
        ## Calculate pearson correlation
        return loss.mean(), (pred, updated_state)
    
    (loss, (pred, updated_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics_updates = state.metrics.single_from_model_output(loss=loss, logits=pred, labels=label)
    metrics = state.metrics.merge(metrics_updates)
    state = state.replace(metrics=metrics)
    state = state.replace(state=updated_state)
    if return_grads: return state, grads
    else: return state

# %%
@jax.jit
def compute_metrics(*, state, batch):
    """Obtaining the metrics for a given batch."""
    img, label = batch
    def loss_fn(params):
        ## Forward pass through the model
        pred = state.apply_fn({"params": params, **state.state}, img, train=False)

        ## Calculate the distance
        loss = optax.softmax_cross_entropy_with_integer_labels(pred, label)
        
        ## Calculate pearson correlation
        return loss.mean(), pred
    (loss, pred) = loss_fn(state.params)
    metrics_updates = state.metrics.single_from_model_output(loss=loss, logits=pred, labels=label)
    metrics = state.metrics.merge(metrics_updates)
    state = state.replace(metrics=metrics)
    return state

# %%
state = create_train_state(Model(config), random.PRNGKey(config.SEED), optax.adam(config.LEARNING_RATE), input_shape=(1,*next(iter(dst_train))[0].shape))

# %%
if config.USE_DNGAUSS:
    params = unfreeze(state.params)
    params["GDNGaussian_0"]["GaussianLayerGamma_0"]["gamma"] = random.uniform(key=random.PRNGKey(config.SEED), shape=params["GDNGaussian_0"]["GaussianLayerGamma_0"]["gamma"].shape)
    params["GDNGaussian_1"]["GaussianLayerGamma_0"]["gamma"] = random.uniform(key=random.PRNGKey(config.SEED), shape=params["GDNGaussian_1"]["GaussianLayerGamma_0"]["gamma"].shape)
    state = state.replace(params=freeze(params))

# %%
jax.tree_util.tree_map(lambda x: x.shape, state.params)

# %%
param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
# trainable_param_count = sum([w.size if t=="trainable" else 0 for w, t in zip(jax.tree_util.tree_leaves(state.params), jax.tree_util.tree_leaves(trainable_tree))])
print(f"Total parameters: {param_count}")#, trainable_param_count

# %%
wandb.run.summary["total_parameters"] = param_count

# %%
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(state)

# %%
metrics_history = {
    "train_loss": [],
    "train_accuracy": [],
}
if dst_val is not None:
    metrics_history["val_loss"] = []
    metrics_history["val_accuracy"] = []

# %%
from functools import partial

# %%
batch = next(iter(dst_train_rdy.as_numpy_iterator()))

# %%
@jax.jit
def forward(state, inputs):
    return state.apply_fn({"params": state.params, **state.state}, inputs, train=False)

# %%
@jax.jit
def forward_intermediates(state, inputs):
    return state.apply_fn({"params": state.params, **state.state}, inputs, train=False, capture_intermediates=True)

# %%
 
outputs = forward(state, batch[0])
outputs.shape

# %%
 
s1, grads = train_step(state, batch, return_grads=True)

# %%
 
for epoch in range(config.EPOCHS):
    ## Training
    for batch in dst_train_rdy.as_numpy_iterator():
        state, grads = train_step(state, batch, return_grads=True)
        wandb.log({f"{k}_grad": wandb.Histogram(v) for k, v in flatten_params(grads).items()}, commit=False)
        # state = compute_metrics(state=state, batch=batch)
        # break

    ## Log the metrics
    for name, value in state.metrics.compute().items():
        metrics_history[f"train_{name}"].append(value)
    
    ## Empty the metrics
    state = state.replace(metrics=state.metrics.empty())

    ## Evaluation
    if dst_val is not None:
        for batch in dst_val_rdy.as_numpy_iterator():
            state = compute_metrics(state=state, batch=batch)
            # break
        for name, value in state.metrics.compute().items():
            metrics_history[f"val_{name}"].append(value)
        state = state.replace(metrics=state.metrics.empty())
    
    ## Obtain activations of last validation batch
    _, extra = forward_intermediates(state, batch[0])

    ## Checkpointing
    if dst_val is not None:
        if metrics_history["val_loss"][-1] <= min(metrics_history["val_loss"]):
            orbax_checkpointer.save(os.path.join(wandb.run.dir, "model-best"), state, save_args=save_args, force=True) # force=True means allow overwritting.

    wandb.log({"gammax": state.params["GaborReductionReLUBlock_0"]["GaborLayerGamma__0"]["gammax"],
                    "mean_gammax": state.params["GaborReductionReLUBlock_0"]["GaborLayerGamma__0"]["gammax"].mean(),
                    "mean_gammax": state.params["GaborReductionReLUBlock_0"]["GaborLayerGamma__0"]["gammax"].std(),
                    "gammay": state.params["GaborReductionReLUBlock_0"]["GaborLayerGamma__0"]["gammay"].mean(),
                    "mean_gammay": state.params["GaborReductionReLUBlock_0"]["GaborLayerGamma__0"]["gammay"].mean(),
                    "mean_gammay": state.params["GaborReductionReLUBlock_0"]["GaborLayerGamma__0"]["gammay"].std()})
    wandb.log({f"{k}": wandb.Histogram(v) for k, v in flatten_params(state.params).items()}, commit=False)
    wandb.log({f"{k}": wandb.Histogram(v) for k, v in flatten_params(extra["intermediates"]).items()}, commit=False)
    wandb.log({"epoch": epoch+1, **{name:values[-1] for name, values in metrics_history.items()}})
    if dst_val is not None:
        print(f'Epoch {epoch} -> [Train] Loss: {metrics_history["train_loss"][-1]} | Acc: {metrics_history["train_accuracy"][-1]} [Val] Loss: {metrics_history["val_loss"][-1]} | Acc: {metrics_history["val_accuracy"][-1]}')
    else:
        print(f'Epoch {epoch} -> [Train] Loss: {metrics_history["train_loss"][-1]} | Acc: {metrics_history["train_accuracy"][-1]}')
    # break
