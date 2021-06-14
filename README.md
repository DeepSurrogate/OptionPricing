# Deep-Structural-Estimation

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


## Prerequisites / Installation
TensorFlow 2
```shell
$ pip install tensorflow==2.3.1
$ pip install scipy==1.6.3
$ pip install matplotlib==3.4.2
```



# Citation
Please cite Deep Equilibrium Nets in your publications if it helps your research:

Chen, Hui and Didisheim, Antoine and Scheidegger, Simon, Deep Structural Estimation: With an Application to Option Pricing (Mar 12, 2021). Available at SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3782722 or https://arxiv.org/abs/2102.09209
