# AmNat_MS61381R1_ABM

Agent-Based Model software (Python) for AmNat MS 61381R1

## Software requirements:

Python >= 3.9<br>
Python libraries: heapq, numpy, progress, pandas  

## Running the simulation model:

- Modify `input.csv` (see below) to set up initial parameter values for the simulation.
- From a terminal prompt, execute:

  > `python simulation.py input.csv`
  
  which will use `input.csv` to pull initial parameter values, and will by default show a helpful progress bar.
  
  As another example, you can use a different input file (which must follow the format of the provided `input.csv`) and not show the progress bar by providing a different input CSV filename and indicating `False` (not default) for progress bar display:
  
  > `python simulation.py other_input.csv False`
  
- Time-series output of the number of algal symbionts per day (total and per-clade) will appear in an output file whose name is specified using `POPULATION_FILENAME` inside `input.csv`.
- If selected (by setting `WRITE_CSV_INFO` to `True` in `input.csv`), per-symbiont statistical information will be written to a CSV file whose name is specified using `CSV_FILENAME` inside `input.csv`.

## Description of ABM software files:

- `clade.py`

  > - `Clade` class to implement/store clade-level specific values, i.e., values that all symbionts of a particular clade will have.
  
- `event_list.py`

  > - `Event` class to implement an event (to be stored in the corresponding event list) in the simulation model.  An event is determined by its time, the type of event, and the corresponding algal symbiont driving the event.
  > - `EventList` class to implement an event list for the simulation model, storing future events in time-sequenced order.  This uses Python's heapq.heappush and heapq.heappop to efficiently maintain a priority queue of events.

- `input.csv`

  > - CSV (comma-separated value) spreadsheet file containing initial values for simulation-level parameters and for clade-specific parameters.
  > - Lines of the CSV file that begin with # are ignored (as comments).  Blank lines (empty entries) are also ignored.
  > - The file consists of three columns, in order: parameter name, parameter value, full parameter description.
  > - Parameter names in the file match class-level and instance variable names in the software, so **do not alter parameter names in the input file**.

- `parameters.py`

  > - `Parameters` class to allow global-to-the-simulation parameters to be acccessed directly from across the ABM.  These values will be updated by        `parser.py` when parameter values are read from the input CSV.

- `parser.py`

  > - `Parser` class for parsing simulation input parameters that are provided in the CSV input file.

- `rng_mt19937.py`

  > - `RNG` class that implements a wrapper around numpy's MT19937 (Mersenne twister) generator to allow for a "multiple-streams" implementation, i.e., providing a different stream of random numbers for each different stochastic component in the model.  
        
- `simulation.py`

  > - `Simulation` class to implement initialization of the agent-based simulation model and containing the event-driven loop driving the simulation.
  > - This file contains the main function that is the primary point of entry (execution) of the model.

- `sponge.py`

  > - `Cell` class to model a single host cell, having a (row,col) position in a 2D grid of host cells, and able to provide occupancy for an algal symbiont wil requiring a cell-specific photosynthetic demand.
  > - `Sponge` class that implements the 2D sponge environment for a collection of host cells.
  
- `symbiont.py`

  > - `Symbiont` class to implement an algal symbiont in the agent-based simulation.
