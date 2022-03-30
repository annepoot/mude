# annepoot/mude
Jupyter Notebooks for the MUDE course

Each notebook will corresponds to a section in the slides Iuri Rocha has prepared for his part of the MUDE course. 7 notebooks are created, covering the following subjects:

1. k-nearest neighbors
2. Decision theory
3. Generalized linear regression
4. Maximum likelihood estimation
5. Ridge regression
6. Artificial neural networks

# mude_tools
The mude_tools.py file will/can be imported by all notebooks, to increase the amount of functionality in the notebooks without bloating the code. So far only the following class has been implemented:

- magicplotter

## magicplotter

### Description

The `magicplotter` is designed to generate interactive plots of our regression models. Given a data generation function `f_data`, a true function `f_truth` and a prediction function `f_pred`, it will create a plot showing the truth, noisy data and the fitted prediction.

To this plot, arbitrary horizontal and vertical sliders can be added. The variable name that is passed when adding the slider will be used as the key word that is passed to the three


### Attributes

```
inline code
```

### Example



