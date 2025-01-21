# Large Deviation Analysis for Hypothesis Testing for Score based Models
[arXiv] This is an implementation of [Large Deviation Analysis for Hypothesis Testing for Score based Models](https://arxiv.org/abs/2401.15519)


## Requirements
See `requirements.txt`

## Instructions
 - Experimental control are configured in `config.yml`
 - Use `make.sh` to generate run script with `make.py`
 - Use `make.py` to generate exp script to `scripts`
 - Use `make_dataset.py` to prepare datasets
 - Use `process.py` to process exp results
 - Experimental setup are listed in `make.py` 
 - Hyperparameters can be found in `config.yml` and `process_control()` of `module/hyper.py`
 
## Examples
 - Test of Multivariate Normal (MVN) distribution with pertubation $\sigma_{ptb} = 0.02$ on $\mu$  for theoretical limit
    ```ruby
    python test_ht.py --control_name MVN_mvn_lrt-t_0.02-0.0_1
    ```
 - Test of KDDCUP dataset (KDDCUP99) with "back" adversarial network traffic on $W$ of Gauss-Benoulli RBM for empirical limit ($N=10$)
    ```ruby
    python test_ht.py --control_name KDDCUP99_rbm_hst-t_back_1_10
    ```

## Results
- Large deviation analysis of likelihood-based and sore-based hypothesis testing for multivariate normal distribution with perturbation on $\mu$ and $\sigma_{ptb} = 0.02$.
<p align="center">
<img src="/asset/result.png">
</p>

## Acknowledgements
*Enmao Diao  
Taposh Banerjee  
Vahid Tarokh*
