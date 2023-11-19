from cell import *

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
