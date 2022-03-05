import numpy as np
import torch
import json
import yaml
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from omegaconf import DictConfig
import time
torch.set_num_threads(1)

config_path = 'model_config.yaml'
with open(config_path, encoding='utf-8') as f:
    params = yaml.safe_load(f)
preprocessor = EncDecCTCModelBPE.from_config_dict(DictConfig(params).preprocessor)
preprocessor.train(False)

with open("test.f32","rb") as f:
    data = np.fromfile(f, dtype=np.float32)
length = torch.Tensor([np.shape(data)[-1]])    
signal = np.expand_dims(data, axis=0)

start = time.time()
tmp = []
for i in range(1000):
    data, _length = preprocessor(input_signal=torch.from_numpy(signal), length=length)
    tmp.append(data)
end = time.time()
print(end - start)
print(len(tmp))

data = data[0].numpy()
print(data.shape)
with open("out.f32","wb") as f:
    data.tofile(f)
print(length)
print(data)