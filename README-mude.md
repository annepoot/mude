# Installation Instructions
This document contains the instructions to get this week's notebooks up and running.

## Installing Anaconda
We're going to be using Anaconda to create an environment for MUDE, so the first step is to install Anaconda.

[Here](https://www.anaconda.com/products/distribution), you can find the installer for your OS.

## Creating an environment from a .yml file

The second step is to create an environment in Anaconda containing all the required packages. For this purpose, an environment file has been created (`environment.yml`), which tells Anaconda which versions of which python libraries to install. 

You can create your MUDE environment by opening your Anaconda prompt or terminal, and entering
```
conda env create -f environment.yml
```
After waiting for a bit to let Anaconda figure out all the dependencies, installation should follow. You can check that installation was successful by running 
```
conda env list
```
and verifying that `mude` is in the list.

## Activating the MUDE environment
Now, all that's left is to activate the environment before using it, which is done by running
```
conda activate mude
```
Note that by default, no environment is activated, so if you restart your PC, you need to activate the environment again. 

Deactivating the MUDE environment is done by running
```
conda deactivate
```

## Installing additional libraries
In case there are some additional libraries that you would like to install, make sure to do this installation via Anaconda! Packages can be installed by running
```
conda activate mude
conda install <package name>
```
If you don't know the name of your package (this is not necessarily the same name as the one in the import statement in your notebooks/python scripts), you can find it by searching the package you want in the [Anaconda repo](https://anaconda.org/anaconda/repo).

