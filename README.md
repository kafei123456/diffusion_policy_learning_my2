# Diffusion Policy
This repository is a simple PyTorch implementation for paper [**Diffusion Policy:Visuomotor Policy Learning via Action Diffusion**](https://diffusion-policy.cs.columbia.edu/)

## Download example data
```console
import os
import gdown
if not os.path.isfile(dataset_path):
    id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
    gdown.download(id=id, output=dataset_path, quiet=False)
```
## Training
```console
python main.py --config-name=test
```

## Citation
Thanks for the following excellent work：
```
https://diffusion-policy.cs.columbia.edu/
```
