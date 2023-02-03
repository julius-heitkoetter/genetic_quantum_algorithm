# genetic_quantum_algorithm
A comparison between quantum and classical neural networks through the lens of a genetic algorithm simulation.

To run the code, simply run:

``` 
python3 final_quantum_genetic.py 65
```

and replace the 65 with your own number to serve as the random seed (inserted for reproducibility).

To modify the simulation, go into `final_quantum_genetic.py` and modify the "Parameters" section. Most parameters can be freely adjusted. Note that currently `ANIMATION_LENGTH=1000`, which with the current configuration can take up to 30-60 minutes to generate the outputs on a standard processor. 

Additionally, currently the outputs are directed towards an `output` directory. Make sure this exists or redirect where the output files are saved in the last few lines of the program. 
