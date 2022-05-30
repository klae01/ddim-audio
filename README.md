# Audio Generation based on Denoising Diffusion Implicit Models (DDIM)

diffusion model for Audio

<a href="http://www.youtube.com/watch?v=MkLm7c4dP3Y" target="_blank">![](http://img.youtube.com/vi/MkLm7c4dP3Y/0.jpg)</a>  <a href="http://www.youtube.com/watch?v=EVYUzKzOHQ4" target="_blank">![](http://img.youtube.com/vi/EVYUzKzOHQ4/0.jpg)</a>  <a href="http://www.youtube.com/watch?v=fHV6P9srrCA" target="_blank">![](http://img.youtube.com/vi/fHV6P9srrCA/0.jpg)</a>

## Prerequests
Pytorch  Transformers


## Train a model
```
python3 main.py --config audio.yml --doc "test" --ni
```

## Sampling from the model


### Sampling from the sequence of audio that lead to the sample

You can edit the config file to adjust the `num_samples` and length(`t_size`). \
See `sampling` in the config file

Use `--sequence {number of intermediates}` option.

If you want to get all samples, `--sequence -1` or `--sequence 0`
```shell
python3 main.py --config audio.yml --doc "test" --sample --sequence 10 --timesteps 1000 --ni
```


## References and Acknowledgements
```
@article{song2020denoising,
  title={Denoising Diffusion Implicit Models},
  author={Song, Jiaming and Meng, Chenlin and Ermon, Stefano},
  journal={arXiv:2010.02502},
  year={2020},
  month={October},
  abbr={Preprint},
  url={https://arxiv.org/abs/2010.02502}
}
```
