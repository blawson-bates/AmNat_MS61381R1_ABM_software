#####################################
class Parameters:
    ''' Class to allow global-to-the-simulation parameters to be acccessed
        directly from across the ABM.  These values will be updated by
        parser.py when parameter values are read from input CSV.
    '''

    global INFINITY
    INFINITY = float('inf')

    INITIAL_SEED:              int         = 0
    MAX_SIMULATED_TIME:        float       = 0.0    # in days 
    NUM_ROWS:                  int         = 0
    NUM_COLS:                  int         = 0 
    NUM_INITIAL_SYMBIONTS:     int         = 0
    INITIAL_PLACEMENT:         'Placement' = None   # 'randomize', 'vertical', or 'horizontal'
    HOST_CELL_DEMAND:          float       = 0.0
    HCD_FUZZ:                  float       = 0.0
    AVG_TIME_BETWEEN_ARRIVALS: float       = 0.0

    NUM_CLADES:                int         = 0
    CLADE_PROPORTIONS:         list[float] = []

    POPULATION_FILENAME:       str         = ""
    WRITE_CSV_INFO:            bool        = False
    CSV_FILENAME:              str         = ""
    WRITE_LOGGING_INFO:        bool        = False
    LOG_FILENAME:              str         = ""

    NUM_ARRIVAL_STRATEGIES:    int         = 0
    ARRIVAL_STRATEGY_PROBS:    list[float] = []
    NUM_DIVISION_STRATEGIES:   int         = 0
    DIVISION_STRATEGY_PROBS:   list[float] = []

    @classmethod
    def printParameters(cls) -> None:
        ''' class-level method to print out values of simulation-level
            parameters '''
        print(f"INITIAL_SEED:              {cls.INITIAL_SEED}")
        print(f"MAX_SIMULATED_TIME:        {cls.MAX_SIMULATED_TIME}")
        print(f"NUM_ROWS:                  {cls.NUM_ROWS}")
        print(f"NUM_COLS:                  {cls.NUM_COLS}")
        print(f"HOST_CELL_DEMAND:          {cls.HOST_CELL_DEMAND}")
        print(f"HCD_FUZZ:                  {cls.HCD_FUZZ}")
        print(f"AVG_TIME_BETWEEN_ARRIVALS: {cls.AVG_TIME_BETWEEN_ARRIVALS}")
        print(f"NUM_CLADES:                {cls.NUM_CLADES}")
        print(f"CLADE_PROPORTIONS:         {cls.CLADE_PROPORTIONS}")
        print(f"CSV_FILENAME:              {cls.CSV_FILENAME}")
        print(f"POPULATION_FILENAME:       {cls.POPULATION_FILENAME}")
        print(f"WRITE_LOGGING_INFO:        {cls.WRITE_LOGGING_INFO}")
        print(f"LOG_FILENAME:              {cls.LOG_FILENAME}")
        print(f"NUM_ARRIVAL_STRATEGIES:    {cls.NUM_ARRIVAL_STRATEGIES}")
        print(f"ARRIVAL_STRATEGY_PROBS:    {cls.ARRIVAL_STRATEGY_PROBS}")
        print(f"NUM_DIVISION_STRATEGIES:   {cls.NUM_DIVISION_STRATEGIES}")
        print(f"DIVISION_STRATEGY_PROBS:   {cls.DIVISION_STRATEGY_PROBS}")
