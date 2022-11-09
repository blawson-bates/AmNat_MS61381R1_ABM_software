from parameters import *
from rng_mt19937 import *

################################################################################
class Cell:
    ''' class to model a single host cell, having a (row,col) position in a 2D
    grid of host cells, and able to provide occupancy for an algal symbiont wil
    requiring a cell-specific photosynthetic demand.
    '''

    __slots__ = ('_row', \
                 '_col', \
                 '_demand', \
                 '_occupied', \
                 '_symbiont', \
                 '_last_occupied_time', 
                 '_sum_residence_time', 
                 '_num_occupants')

    ###############################################
    def __init__(self, row: int, col: int) -> None:
        ''' initializer method for a host cell object
        Parameters:
            row: integer valued row number in [0,num_rows - 1]
            col: integer valued column number in [0,num_cols - 1]
        '''
        self._row      : int        = row
        self._col      : int        = col
        self._demand   : float      = self.computeDemand()
        self._occupied : bool       = False
        self._symbiont : 'Symbiont' = None   # null

        # used to track observation-persistent and time-persistent statistics of 
        # residence time per cell (and eventually, in simulation.py, per row);
        # Note that:
        #      _sum_residence_time / MAX_T  = prop. of time cell is occupied
        #      _sum_residence_time / # occupants = avg time per symbiont occupation
        self._last_occupied_time = INFINITY
        self._sum_residence_time = 0
        self._num_occupants = 0

    ################################
    ''' simple accessors/getters '''
    def getDemand(self)   -> float:              return self._demand
    def getRowCol(self)   -> tuple[int,int]:     return (self._row, self._col)
    def getSymbiont(self) -> 'Symbiont' or None: return self._symbiont
    def isOccupied(self)  -> bool:               return self._occupied

    ######################################################
    def removeSymbiont(self, current_time: float) -> None:
        ''' method to remove the currently occuping symbiont from this host cell
        Parameters:
            current_time: the current time (as a float)
        '''
        self._symbiont = None
        self._occupied = False
        # add to the residence time for this cell
        assert(self._last_occupied_time != INFINITY)
        self._sum_residence_time += (current_time - self._last_occupied_time)
        self._num_occupants += 1
        self._last_occupied_time = INFINITY

    #########################################################################
    def setSymbiont(self, symbiont: 'Symbiont', current_time: float) -> None:
        ''' updates this cell to have a new occupying symbiont at time t
        Parameters:
            symbiont: a Symbiont object
            current_time: the current time (as a float)
        '''
        # it could be the case that we are swapping in a child symbiont
        # and evicting a parent, without ever calling removeSymbiont();
        # if that is the case, make sure to add in the unit-time rectangle
        # corresponding to the parent's time in the cell...
        if self._symbiont is not None:  # something was just evicted...
            self._sum_residence_time += (current_time - self._last_occupied_time)
            self._num_occupants += 1

        self._symbiont = symbiont
        self._occupied = True
        self._last_occupied_time = current_time   # new symbiont's residence starts now

    #################################
    def computeDemand(self) -> float:
        ''' compute photosynthetic demand expected by this host cell per unit time
        Returns:
            the photosynthetic demand required by this host cell (as a float)
        '''
        ## 12 Apr 2016
        # rather than fuzzing uniformly, use Normal with 95% of the data b/w 
        # (mu +/- mu*f) -- see implementation in rng.py
        m = Parameters.HOST_CELL_DEMAND
        f = Parameters.HCD_FUZZ  # assume to be % of the mean
        demand = RNG.fuzz(m, f, Stream.HOST_CELL_DEMAND)

        return demand 

    #########################
    def __str__(self) -> str:
        ''' str version of this Cell object, with row, col, demand, and symbiont
        Returns:
            this Cell object as a str
        '''
        symbiont_id = None if self._symbiont is None else self._symbiont.getID()
        return f"({self._row},{self._col}): demand: {round(self._demand, 3)}" + \
               f"\tsymbiont: {symbiont_id}"

################################################################################
class Sponge:
    ''' This class implements the 2D sponge environment for a collection of host 
        cells.  Essentially, this class is nothing more than a wrapper for a 2D
        list of Cell references.
    '''

    __slots__ = ('_num_rows', '_num_cols', '_cells')

    def __init__(self, num_rows: int, num_cols: int) -> None:
        ''' initializer for a Sponge object
        Parameters:
            num_rows: integer number of rows in the 2D matrix of cells
            num_cols: integer number of columns
        '''
        self._num_rows = num_rows
        self._num_cols = num_cols

        # assign the 2D list of Cell references 
        self._cells = [[Cell(r,c) for c in range(num_cols)] for r in range(num_rows)]

    def getDimensions(self) -> tuple[int, int]:
        ''' returns the sponge dimensions
        Returns:
            a tuple containing the integer number of rows and columns
        '''
        return (self._num_rows, self._num_cols)

    def getCell(self, row: int , col: int) -> Cell:
        ''' returns the Cell object at the given row and column
        Parameters:
            row: integer valued row of desired cell, in [0, num_rows - 1]
            col: integer valued columnof desired cell, in [0, num_cols - 1]
        Returns:
            Cell object @ (row,col)
        Raises:
            ValueError if either the row or col values are out of bounds
        '''
        if row < 0 or row >= self._num_rows or col < 0 or col >= self._num_cols:
            raise ValueError(f"Error in Sponge.getCell: ({row},{col}) out of bounds")
        cell = self._cells[row][col]
        return cell
