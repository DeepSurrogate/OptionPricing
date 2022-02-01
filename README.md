# Deep Surrogate for Asset Pricing
![alt text](https://github.com/DeepSurrogateEstimation/OptionPricing/blob/main/example/example.png)
The figure above the implied volatility surface of the Bates model with Double Jump Exponential generated with QuantLib (left) and the Deep-Surrogate (right)


## The full paper can be found here
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3782722

https://arxiv.org/abs/2102.09209

## Authors

* Hui Chen (MIT Sloan School of Management, huichen@mit.edu)
* Antoine Didisheim (Swiss Finance Institute, antoine.didisheim@unil.ch)
* Simon Scheidegger (Department of Economics, HEC Lausanne, simon.scheidegger@unil.ch)

## Instruction
* Download th git repository and set it in your project working directory. 
* In your python code, import the surrogate class with: *from source.deepsurrogate import DeepSurrogate*
* Define model_name *'heston'* or *'bdjm'* to load pre-trained surrogate. 
* Instantiate a surrogate object with:  *surrogate = DeepSurrogate(model_name)*
* Use *get_iv* to get the model's implied volatility, or *get_iv_delta* to get the first derivative of the model's ivs for each input: 
    * *surogate.get_iv(X)*
    * *surogate.get_iv_delta(X)*
* Use *get_price* to get the model's price, or *get_price_delta* to get the first derivative of the model's prices for each input: 
    * *surogate.get_price(X)*
    * *surogate.get_price_delta(X)* 
* The Input X should be a pandas DataFrame containing the name of the models parameters. Or a numpy with the columns in thee order below:
    * heston | ['strike', 'rf', 'dividend', 'v0', 'T', 'kappa', 'theta', 'sigma', 'rho', 'S']
    * bdjm |  ['strike', 'rf', 'dividend', 'v0', 'T', 'kappa', 'theta', 'sigma', 'rho', 'lambda_parameter', 'nuUp', 'nuDown', 'p', 'S']

## Example

demo.ipynb, test_delta.py, test_delta_iv.py, test_iv.py and test_price.py illustrate how to easily use the library. 

## Parameter range

Surrogate models are defined inside some specific range of parameters. Both models in this surrogate library have been trained inside the range defined in the table below. The surrogate can not price an option with parameters outside of this range of parameters. 

| Parameter| Min | Max  |
| --------- |:------:| ------:|
| T      | 1 | 380 |
| M      | -9.0 | 5.0 |
| rf      | 0.0      |   0.075 |
| d      | 0.0      |   0.075 |
| v_t| 0.01 | 0.90 |
| kappa| 0.1 | 50.0 |
| sigma| 0.1 | 5.0 |
| rho| -1.0 | 0.0 |
| theta | 0.1 | 0.9 |
| lambda | 0.05 | 4.0 |
| nuUp | 0.0 | 0.4 |
| nuDown | 0.0 | 0.4 |
| p | 0.0 | 1.0 |

Where 

<a href="https://www.codecogs.com/eqnedit.php?latex=m=\frac{\ln&space;\left(\hat{K}&space;/&space;F_{T}\right)}{\sqrt{T}&space;0.19}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m=\frac{\ln&space;\left(\hat{K}&space;/&space;F_{T}\right)}{\sqrt{T}&space;0.19}" title="m=\frac{\ln \left(\hat{K} / F_{T}\right)}{\sqrt{T} 0.19}" /></a>

and, 

<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{K}=100&space;\frac{K}{S}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{K}=100&space;\frac{K}{S}" title="\hat{K}=100 \frac{K}{S}" /></a>

## Prerequisites / Installation
TensorFlow 2
```shell
$ pip install tensorflow==2.3.1
$ pip install scipy==1.6.3
$ pip install matplotlib==3.4.2
```

# Support
This work is generously supported by grants from the Swiss Platform for Advanced Scientific
Computing (PASC) under project ID “Computing equilibria in heterogeneous agent macro models on con-
temporary HPC platforms”, the Swiss National Supercomputing Center (CSCS) under project ID 995, and
the Swiss National Science Foundation under project ID “New methods for asset pricing with frictions”.


# Citation
Please cite Deep Surrogate Estimation: With an Application to Option Pricing in your publications if it helps your research:

Chen, Hui and Didisheim, Antoine and Scheidegger, Simon, Deep Structural Estimation: With an Application to Option Pricing (Mar 12, 2021). Available at SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3782722 or https://arxiv.org/abs/2102.09209
```
@article{chen2021deep,
  title={Deep Structural Estimation: With an Application to Option Pricing},
  author={Chen, Hui and Didisheim, Antoine and Scheidegger, Simon},
  journal={Available at SSRN 3782722},
  year={2021}
}
```
