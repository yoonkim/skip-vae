# Semi-Amortized Variational Autoencoders
Code for the paper:  
[Avoiding Latent Variable Collapse With Generative Skip Models](https://arxiv.org/pdf/1807.04863.pdf)
Adji B. Dieng, Yoon Kim, Alexander M. Rush, David M. Blei.

Our code/data is based on the [Semi-Amortized VAE repo](https://github.com/harvardnlp/sa-vae).
Please refer to the above repo for dependencies, data processing, etc.

## Model
After downloading the `sa-vae` repo, copy these files to the `sa-vae` folder:  
- `train_text_skip.py`
- `models_text_skip.py`
- `train_img_skip.py`
- `models_img_skip.py`

To run the text model:
```
python train_text_skip.py --train_file data/yahoo/yahoo-train.hdf5 --val_file data/yahoo/yahoo-val.hdf5 --gpu 1 --checkpoint_path model-path --skip 1 --model savae --svi_steps 20 --train_n2n 1
```
where `model-path` is the path to save the best model and the `*.hdf5` files are obtained from running `preprocess_text.py`. You can specify which GPU to use by changing the input to the `--gpu` command.

To run the image model:
```
python train_img_skip.py --data_file data/omniglot/omniglot.pt --gpu 1 --checkpoint_path model-path --skip 1 --model savae --svi_steps 20 --train_n2n 1
```
## License
MIT