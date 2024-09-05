# importing libraries and modified ParticleTransformer class for quantization

import numpy as np
import awkward as ak
import uproot
import vector
vector.register_awkward()
import os
import shutil
import zipfile
import tarfile
import urllib
import requests
from tqdm import tqdm
import torch
import timeit
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from ParticleTransformer_updated import ParticleTransformer
from ParticleTransformer_updated_quant_weights import ParticleTransformer as ParticleTransformer_quant
from weaver.utils.logger import _logger
import torch.optim as optim
import time

# data pre processing functions

def build_features_and_labels(tree, transform_features=True):
    # load arrays from the tree
    a = tree.arrays(filter_name=['part_*', 'jet_pt', 'jet_energy', 'label_*'])

    # compute new features
    a['part_mask'] = ak.ones_like(a['part_energy'])
    a['part_pt'] = np.hypot(a['part_px'], a['part_py'])
    a['part_pt_log'] = np.log(a['part_pt'])
    a['part_e_log'] = np.log(a['part_energy'])
    a['part_logptrel'] = np.log(a['part_pt']/a['jet_pt'])
    a['part_logerel'] = np.log(a['part_energy']/a['jet_energy'])
    a['part_deltaR'] = np.hypot(a['part_deta'], a['part_dphi'])
    a['part_d0'] = np.tanh(a['part_d0val'])
    a['part_dz'] = np.tanh(a['part_dzval'])

    # apply standardization
    if transform_features:
        a['part_pt_log'] = (a['part_pt_log'] - 1.7) * 0.7
        a['part_e_log'] = (a['part_e_log'] - 2.0) * 0.7
        a['part_logptrel'] = (a['part_logptrel'] - (-4.7)) * 0.7
        a['part_logerel'] = (a['part_logerel'] - (-4.7)) * 0.7
        a['part_deltaR'] = (a['part_deltaR'] - 0.2) * 4.0
        a['part_d0err'] = _clip(a['part_d0err'], 0, 1)
        a['part_dzerr'] = _clip(a['part_dzerr'], 0, 1)

    feature_list = {
        'pf_points': ['part_deta', 'part_dphi'], # not used in ParT
        'pf_features': [
            'part_pt_log',
            'part_e_log',
            'part_logptrel',
            'part_logerel',
            'part_deltaR',
            'part_charge',
            'part_isChargedHadron',
            'part_isNeutralHadron',
            'part_isPhoton',
            'part_isElectron',
            'part_isMuon',
            'part_d0',
            'part_d0err',
            'part_dz',
            'part_dzerr',
            'part_deta',
            'part_dphi',
        ],
        'pf_vectors': [
            'part_px',
            'part_py',
            'part_pz',
            'part_energy',
        ],
        'pf_mask': ['part_mask']
    }

    out = {}
    for k, names in feature_list.items():
        out[k] = np.stack([_pad(a[n], maxlen=128).to_numpy() for n in names], axis=1)

    label_list = ['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q', 'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']
    out['label'] = np.stack([a[n].to_numpy().astype('int') for n in label_list], axis=1)

    return out

def _clip(a, a_min, a_max):
    try:
        return np.clip(a, a_min, a_max)
    except ValueError:
        return ak.unflatten(np.clip(ak.flatten(a), a_min, a_max), ak.num(a))

def _pad(a, maxlen, value=0, dtype='float32'):
    if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
        return a
    elif isinstance(a, ak.Array):
        if a.ndim == 1:
            a = ak.unflatten(a, 1)
        a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
        return ak.values_astype(a, dtype)
    else:
        x = (np.ones((len(a), maxlen)) * value).astype(dtype)
        for idx, s in enumerate(a):
            if not len(s):
                continue
            trunc = s[:maxlen].astype(dtype)
            x[idx, :len(trunc)] = trunc
        return x

# Quantizable ParticleTransformer model


class ParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = ParticleTransformer_quant(**kwargs)
        self.attention_matrix = None
        self.interactionMatrix = None
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask):
        output = self.mod(features, v=lorentz_vectors, mask=mask)
        self.attention_matrix = self.mod.getAttention()
        self.interactionMatrix = self.mod.getInteraction()
        return output

    def get_attention_matrix(self):
        return self.attention_matrix
    def get_interactionMatrix(self):
        return self.interactionMatrix

def get_model(**kwargs):

    cfg = dict(
        input_dim=17,
        num_classes=10,
        # network configurations
        pair_input_dim=4,
        use_pre_activation_pair=False,
        embed_dims=[128, 512, 128],
        pair_embed_dims= [64,64,64],
        num_heads=8,
        num_layers=8,      # make it 8
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        # misc
        trim=True,
        for_inference=False,
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = ParticleTransformerWrapper(**cfg)

    model_info = {
    }

    return model, model_info

quantizable_model, _ = get_model()

pretrained_dict = torch.load("ParT_full.pt", map_location=torch.device('cpu'))
def adapt_weights(pretrained_dict):
    new_dict = {}
    for key, value in pretrained_dict.items():
        if 'attn.in_proj_weight' in key:
            # Split the original in_proj_weight into Q, K, V
            q, k, v = torch.chunk(value, 3, dim=0)
            base_key = key.replace('in_proj_weight', 'linear_Q.weight')
            new_dict[base_key] = q
            new_dict[base_key.replace('linear_Q', 'linear_K')] = k
            new_dict[base_key.replace('linear_Q', 'linear_V')] = v
        elif 'attn.in_proj_bias' in key:
            # Split the original in_proj_bias into Q, K, V
            q, k, v = torch.chunk(value, 3, dim=0)
            base_key = key.replace('in_proj_bias', 'linear_Q.bias')
            new_dict[base_key] = q
            new_dict[base_key.replace('linear_Q', 'linear_K')] = k
            new_dict[base_key.replace('linear_Q', 'linear_V')] = v
        else:
            new_dict[key] = value
    return new_dict
adapted_weights = adapt_weights(pretrained_dict)
model_dict = quantizable_model.state_dict()
model_dict.update(adapted_weights)  # Update the model's state dict with the adapted weights
quantizable_model.load_state_dict(model_dict)

# setting the loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Setting the function

quantized_model = torch.quantization.quantize_dynamic(
    quantizable_model, {torch.nn.Linear, torch.nn.LayerNorm}, dtype=torch.qint8
)

# Comparing the model sizes

# creating an iterative dataloader that loads the data

import os
import uproot
from torch.utils.data import IterableDataset, DataLoader
import numpy as np

class IterativeParticleDataset(IterableDataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.root')]
        print(self.files)

    def process_file(self, file_path):
        """Load data from a single .root file and yield it."""
        tree = uproot.open(file_path)['tree']  # Adjust tree name as necessary
        table = build_features_and_labels(tree)  # Define this function based on your data structure

        # Loop over each sample in the file
        for i in range(len(table['pf_features'])):
            yield {
                'x_particles': table['pf_features'][i],
                'x_jets': table['pf_vectors'][i],
                'y': table['label'][i],
                'x_points': table['pf_points'][i],
                'x_mask': table['pf_mask'][i],
            }

    def __iter__(self):
        """Iterate over all files and yield data from each file one by one."""
        for file_path in self.files:
            print(file_path)
            yield from self.process_file(file_path)  # Yield data from each file one sample at a time

# data folder containing the test .root files
folder_path = 'test_20M'
dataset = IterativeParticleDataset(folder_path)

# DataLoader with IterableDataset
dataloader = DataLoader(dataset, batch_size=256, shuffle=False)


# Perform inference in a memory-efficient manner
from tqdm import tqdm

num = 1

def inference_timer(model, dataloader, y_true, y_prob):
    global c        # so that the outputs are loaded just once
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            #print(id(batch), id(dataloader))
            x_particles = batch['x_particles']
            x_jets = batch['x_jets']
            y = batch['y']
            x_points = batch['x_points']
            x_mask = batch['x_mask']

            # Perform inference
            outputs = model(x_points, x_particles, x_jets, x_mask)
            # Store outputs and labels for later loss calculation
            if c < 1:
                y_true.append(y)
                y_prob.append(outputs)
                #print('Appending the true and predicted labels')
    c += 1
    
all_outputs_base = []
all_labels_base = []

c = 0
execution_time_base_test = timeit.timeit(lambda: inference_timer(quantized_model, dataloader, all_labels_base, all_outputs_base), number=num)

all_outputs_base = torch.cat(all_outputs_base, dim=0)
all_labels_base = torch.cat(all_labels_base, dim=0)
np.save('outputs_base', all_outputs_base)
np.save('labels_base', all_labels_base)

base_model_loss = loss_fn(all_outputs_base.float(), all_labels_base.float())

print(base_model_loss)
base_model_loss = loss_fn(all_outputs_base.float(), all_labels_base.float())


