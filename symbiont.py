import logging
import copy  # for copy constructor:
    # http://stackoverflow.com/questions/1241148/copy-constructor-in-python
    # http://pymotw.com/2/copy/
from numpy import cumsum
from parameters import *
from rng_mt19937 import *
from event_list import Event, EventType
from clade import *
from cell import *

###############################################################################
# This class implements a symbiont alga in the agent-based simulation.
# Each symbiont will have the following instance variables, as well as those
# contained by its parent class Clade:
# 
#    self._id                      : integer count of this symbiont
#    self._cell                    : cell of residence (none if evicted)
#    self._mitotic_cost_rate       : cost of mitosis (photosynthate per unit time)
#    self._production_rate         : photosynthetic production rate (per unit time)
#    self._photosynthate_surplus   : banked photosynthate (e.g., to use for mitosis)
#    self._surplus_on_arrival      : photosynthate on arrival (from pool or inherited)
#
#    self._arrival_time            : when this symbiont appeared in the system
#    self._prev_event_time         : last time this symbiont did something
#    self._prev_event_type         : last type of thing this symbiont did 
#    self._time_of_escape          : time when symbiont is exiting quickly
#    self._time_of_digestion       : time when symbiont will be digested
#    self._time_of_denouement      : time when sybmiont leaves on own accord
#    self._time_of_next_end_g0     : time when next G0 period ends
#    self._time_of_next_end_g1sg2m : time when next mitosis ends
#
#    self._next_event_time         : next (min) event time among those above
#    self._next_event_type         : event type associated with the next event
#    self._prev_event_time         : previous (min) event time among those above
#    self._prev_event_type         : event type associated with the previous event
#
#    self._how_arrived             : ARRIVED_FROM_POOL (0), ARRIVED_VIA_DIVISION (1)
#    self._parent_id               : id of parent, -1 if arrived from pool
#    self._agent_zero              : ultimate ancestor (may be self)
#    self._surplus_on_arrival      : photosynthate on hand when arrived
#    self._num_divisions           : number of successful divisions for symbiont
#    self._clade_number            : the number ID of the clade of this symbiont
#    self._my_clade                : an instance of the parent Clade of this symbiont
#
#    self._cells_at_division       : list of number of open cells available at each mitosis
#    self._cells_inhabited         : list of all cells inhabited
#    self._inhabit_times           : list of times each cell above inhabited
#    self._hcds_of_cells_inhabited : list of photosynthate demand of each cell inhabited
#    self._g0_times                : list of g0 times for this symbiont
#    self._g1sg2m_times            : list of g1sg2m times for this symbiont
#    self._num_divisions           : number of successful divisions for this symbiont
#
# Methods of interest (all but one public-facing) used in the simulation:
#    __init__                          : construct a new arrival from pool
#    _SymbiontCopy(cell, current_time) : construct a new arrival from mitotsis
#    endOfG0(t)                        : handle end of G0 period
#    endOfG1SG2M(t)                    : handle end of G1SG2M, when division occurs
#    digestion(t)                      : use when symbiont digestion occurs
#    escape(t)                         : use when symbiont escape occurs
#    denouement(t)                     : use when symbiont denouement occurs
#    __str__()                         : use to print a symbiont
#    getNextEvent()                    : returns next event as (time,EventType)
#    openCSVFile        [class-level]  : open CSV file for writing per-symbiont info (if requested)
#    csvOutputOnExit    [class-level]  : dumps symbiont info to CSV @ time of symbiont exit
#    csvOutputAtEnd     [class-level]  : dumps remaining in-residence symbiont info to CSV @ simulation end
#    findOpenCell       [class-level]  : finds an open cell at random among all avaiable in sponge
#    findOpenCellWithin [class-level]  : finds an open cell at random within a given neighborhood
#    generateArrival    [class-level]  : generates a symbiont arrival, if space in the sponge
#
# Notes on Affinity Terms (Spring 2019):
#   ARRIVAL AFFINITY - the level of coevolution between symbiont and host in 
#   regards to the ability of the symbiont to safely enter the host cell. 
#   Can take the form of a number of different strategies based on different
#   biological features that compose it.
# ----------------------------------------------------------------------------
#   DIVISION AFFINITY - the level of coevolution between symbiont and host in 
#   regards to the ability of the symbiont to safely maintain presense the host 
#   cell. Can take the form of a number of different strategies based on different
#   biological features that compose it.
###############################################################################

################################################################################
class SymbiontState(Enum):
    # used in division cases on border
    CELL_OUTSIDE_ENVIRONMENT = -1
    # to identify Symbiont's arrival method
    ARRIVED_FROM_POOL        =  0
    ARRIVED_VIA_DIVISION     =  1
    # states used for computing surplus at event end
    IN_G0                    =  2
    IN_G1SG2M                =  3
    # states for what happens at end of G1SG2M
    CHILD_INFECTS_OUTSIDE    =  4
    CHILD_EVICTED            =  5
    CHILD_NO_AFFINITY        =  6  # 18 Mar 2017
    PARENT_INFECTS_OUTSIDE   =  7
    PARENT_EVICTED           =  8
    PARENT_NO_AFFINITY       =  9  # 18 Mar 2017
    BOTH_STAY                = 10
    # additional states to identify exit status
    STILL_IN_RESIDENCE       = 11
    DIGESTION_IN_G0          = 12
    DIGESTION_IN_G1SG2M      = 13
    ESCAPE_IN_G0             = 14
    ESCAPE_IN_G1SG2M         = 15
    DENOUEMENT_IN_G0         = 16
    DENOUEMENT_IN_G1SG2M     = 17

################################################################################
class Symbiont:
    # the __slots__ tuple defines the names of the instance variables for a 
    # Symbiont object so that, e.g.,  mistyping a name doesn't accidentally
    # introduce a new instance variable (a Python "feature" if using the 
    # default dict approach without __slots__ defined)
    __slots__ = ( \
                 '_agent_zero',              \
                 '_arrival_time',            \
                 '_cell',                    \
                 '_cells_at_division',       \
                 '_cells_inhabited',         \
                 '_clade_number',            \
                 '_g0_times',                \
                 '_g1sg2m_times',            \
                 '_hcds_of_cells_inhabited', \
                 '_how_arrived',             \
                 '_id',                      \
                 '_inhabit_times',           \
                 '_mitotic_cost_rate',       \
                 '_my_clade',                \
                 '_next_event_time',         \
                 '_next_event_type',         \
                 '_num_divisions',           \
                 '_parent_id',               \
                 '_photosynthate_surplus',   \
                 '_prev_event_time',         \
                 '_prev_event_type',         \
                 '_production_rate',         \
                 '_surplus_on_arrival',      \
                 '_time_of_escape',          \
                 '_time_of_digestion',       \
                 '_time_of_denouement',      \
                 '_time_of_next_end_g0',     \
                 '_time_of_next_end_g1sg2m', \
                 )

    # class-level variables
    sponge : 'Sponge' = None   # set in simulation.py main function when symbionts are created
    clade_cumulative_proportions : list[float] = [None] * len(Parameters.CLADE_PROPORTIONS)
    _count : int      = 0      # used to count total number of symbionts

    # class-level variables for writing per-symbiont statistics; whether to
    # write the CSV information can be changed via command-line argument at
    # simulation execution
    _write_csv:  bool                = False  
    _csv_writes: int                 = 0
    _csv_file:   '_io.TextIOWrapper' = None

    ############################################################################
    def __init__(self, clade_number: int, cell: Cell, current_time: float) -> None:
        ''' initializer used to create symbionts that arrive from the pool
            (oustide)
        Parameters:
            clade_number: integer corresponding to the specific clade of this symbiont
            cell: Cell object into which this symbiont will be arriving
            current_time: floating point value of current simulation time
        Raises:
            RuntimeError, if the Symbiont class-level sponge variable has not been set
            ValueError, if the provided clade number is invalid
        '''
        # Error checking
        if Symbiont.sponge is None:
            raise RuntimeError(f"Error in Symbiont: class-level sponge environment not set")
        if clade_number < 0 or clade_number >= Parameters.NUM_CLADES:
            raise ValueError(f"Error in Symbiont: invalid clade {clade_number}")

        ########################################################################
        # define instance variables and type (hints) so they are in one place
        # for easy reference; actual values will be assigned below and/or later
        self._id:                      int           = None
        self._clade_number:            int           = None
        self._my_clade:                Clade         = None
        self._cell:                    Cell          = None
        self._how_arrived:             SymbiontState = None
        self._parent_id:               int           = None
        self._agent_zero:              int           = None
        self._num_divisions:           int           = None
        self._mitotic_cost_rate:       float         = None
        self._production_rate:         float         = None
        self._photosynthate_surplus:   float         = None
        self._surplus_on_arrival:      float         = None
        self._cells_inhabited:         list[str]     = None
        self._inhabit_times:           list[float]   = None
        self._hcds_of_cells_inhabited: list[float]   = None 
        self._g0_times:                list[float]   = None
        self._g1sg2m_times:            list[float]   = None
        #self._cells_at_division:       list[int]     = None

        self._arrival_time:            float         = None
        self._time_of_escape:          float         = None
        self._time_of_digestion:       float         = None
        self._time_of_denouement:      float         = None
        self._time_of_next_end_g0:     float         = None
        self._time_of_next_end_g1sg2m: float         = None

        self._prev_event_time:         float         = None
        self._prev_event_type:         EventType     = None
        self._next_event_time:         float         = None
        self._next_event_type:         EventType     = None
        ########################################################################

        self._id = Symbiont._count
        Symbiont._count += 1

        self._clade_number  = clade_number
        self._my_clade      = Clade.getClade(clade_number)
        self._cell          = cell
        self._how_arrived   = SymbiontState.ARRIVED_FROM_POOL
        self._parent_id     = -1        # arriving from pool, no parent
        self._agent_zero    = self._id  # arriving from pool, topmost pregenitor is self
        self._num_divisions = 0

        # INDIVIDUAL SYMBIONT FUZZING
        # Rather than fuzzing uniformly, use normal with 95% of the data
        # between (mu +/- mu*f) -- see implementation in rng.py
        m = float(self._my_clade.getMCR())
        f = float(self._my_clade.getMCRFuzz())  # assume to be % of the mean
        self._mitotic_cost_rate = RNG.fuzz(m, f, Stream.MITOTIC_COST_RATE)

        self._production_rate = self._computeProductionRate(is_copy = False, current_time = current_time)

        # 21 Apr 2016: change initial photosynthate to use a Gamma distribution;
        # Connor and Barry experimented and determined gamma(2,0.75) looks
        # reasonable -- mean = 1.5, 50% = 1.25
        # 08 Oct 2016: add max, per Malcolm suggetion in 7 Oct meeting
        clade_max_photosynthate = self._my_clade.getMaxInitialSurplus()
        self._photosynthate_surplus = INFINITY
        while self._photosynthate_surplus > clade_max_photosynthate:
            self._photosynthate_surplus = RNG.gamma( \
                self._my_clade.getInitialSurplusShape(), \
                self._my_clade.getInitialSurplusScale(), \
                Stream.PHOTOSYNTHATE)
        #print(">>>>>>>>> ORIG SURPLUS= ",self._photosynthate_surplus)

        self._surplus_on_arrival = self._photosynthate_surplus

        # 22 Sep 2016: Malcolm wanted to keep track of total residence time per
        # symbiont as well as residence time per cell for a symbiont (e.g., when
        # parent divides and parent moves to a different cell) -- the following
        # data structure is to keep track of times when a symbiont does not exit
        # the system but moves into a different cell within our grid
        #
        # 5 Oct 2016: also want a list of all cells visited so we can dump that
        # info into CSV at end, but more importantly the hcd for those cells
        # (looking for evolutionary advantages given symbionts)...
        self._cells_inhabited         = [str(cell.getRowCol()).replace(', ',',')]
        self._inhabit_times           = [current_time]
        self._hcds_of_cells_inhabited = [cell.getDemand()]

        # 5 Oct 2016: Malcolm also wanted to keep track of the g0 and g1sg2m
        # lengths for symbiont's, to look for evolutionary advantage; these
        # times are created at random with each g0 and g1sg2m event, so we need
        # to also store all of them in a list per symbiont
        self._g0_times = []
        self._g1sg2m_times = []

        # 13 Feb 2017: also track the # of open cells around at time of division
        #self._cells_at_division = []

        self._arrival_time              = current_time  # when arrived -- now!
        self._prev_event_time           = current_time
        self._prev_event_type           = EventType.ARRIVAL
        self._time_of_escape            = INFINITY      # global in Parameters
        self._time_of_digestion         = INFINITY  
        self._time_of_denouement        = INFINITY
        self._time_of_next_end_g0       = INFINITY      # end of G0, start of G1/S/G2/M
        self._time_of_next_end_g1sg2m   = INFINITY      # end of G1/S/G2/M, start of G0

        self._next_event_time           = INFINITY      # set in scheduling events next...
        self._next_event_type           = None

        self._scheduleInitialEvents(current_time)
        self._setNextEvent()

    #############################################################################
    def _SymbiontCopy(self, cell: Cell or None, current_time: float) -> 'Symbiont':
        ''' This "copy constructor" is used to create symbionts that occur from
            mitosis.  The resulting symbiont should have the same clade.  The
            copy will have its own id, own event times, and (depending on
            row-location of the host cell) potentially different photosynthate
            production rate.
        Parameters:
            cell: the Cell object into which the new symbiont will go, or None
                if no cell available or if infecting outside the model's scope
            current_time: floating point value of current simulation time

        Returns:
            a newly copied and modified Symbiont object resulting from mitosis
        '''
        new_symbiont = copy.copy(self)  # make an exact copy of this symbiont

        # now begin updating its values as a symbiont arriving anew from mitosis
        new_symbiont._id = Symbiont._count
        Symbiont._count += 1

        new_symbiont._num_divisions = 0

        new_symbiont._how_arrived     = SymbiontState.ARRIVED_VIA_DIVISION
        new_symbiont._parent_id       = self._id          # new symbiont's parent is this symbiont
        new_symbiont._agent_zero      = self._agent_zero  # same original progenitor as self

        new_symbiont._cell            = cell
        new_symbiont._arrival_time    = current_time  # when arrived -- now!
        new_symbiont._prev_event_time = current_time
        new_symbiont._prev_event_type = EventType.ARRIVAL

        # 23 Sep 2016 and 5 Oct 2016 and 13 Feb 2017:
        # clear out any switched times inherited from parent
        new_symbiont._cells_inhabited         = []  # may be updated below...
        new_symbiont._inhabit_times           = []
        new_symbiont._hcds_of_cells_inhabited = []
        new_symbiont._g0_times                = []
        new_symbiont._g1sg2m_times            = []
        #new_symbiont._cells_at_division       = []

        ## 12 Apr 2016
        # fuzz all of the inherited values (MCR, PPR, surplus)
        # use normal with 95% of the data b/w (mu +/- mu*f) -- see rng.py;
        # MCR is fuzzed here; surplus below; PPR inside _computeProductionRate()
        #####################################################
        assert(new_symbiont._mitotic_cost_rate == self._mitotic_cost_rate)
        m = new_symbiont._mitotic_cost_rate

        #####################################################
        ## OLD WAY -- using normal
        '''
        f = Parameters.DIV_FUZZ  # assume to be % of the mean
        new_symbiont._mitotic_cost_rate = RNG.fuzz(m, f, Stream.MITOTIC_COST_RATE)
        '''
        ## NEW WAY -- using combined gamma for deleterious & beneificial
        [fuzzamt, mutation] = RNG.divfuzz(m, self._my_clade, Stream.MITOTIC_COST_RATE_MUTATION)
        if mutation == MutationType.DELETERIOUS:
            new_symbiont._mitotic_cost_rate += fuzzamt  # deleterious mcr increases
        else:  # mutation == MutationType.BENEFICIAL:
            new_symbiont._mitotic_cost_rate -= fuzzamt  # beneficial mcr decreases

        # uncomment below if want to see info about mutations...
        '''
        if mutation != MutationType.NO_MUTATION:
            mut_type = 'DEL' if mutation == MutationType.DELETERIOUS else 'BEN'
            print(f'@ t={current_time} {new_symbiont._id} {mut_type} mcr: {m} {new_symbiont._mitotic_cost_rate}')
        '''

        # give up roughly half (fuzzed) of the banked photosynthate to new symbiont
        half = self._photosynthate_surplus / 2
        #old = self._photosynthate_surplus  # for testing whether halving works correctly (below)

        ## OLD WAY -- using normal
        '''
        m = half
        f = Parameters.DIV_FUZZ
        half = RNG.fuzz(m, f, Stream.PHOTOSYNTHATE)
        fuzzedhalf = RNG.divfuzz(half, Stream.PHOTOSYNTHATE)
        '''
        ## NEW WAY -- using combined exponential for deleterious & beneficial
        [fuzzamt, mutation] = RNG.divfuzz(half, self._my_clade, Stream.PHOTOSYNTHATE_MUTATION)
        fuzzedhalf = half
        if mutation == MutationType.DELETERIOUS:
            fuzzedhalf -= fuzzamt  # deleterious inheritance slightly less than half
        elif mutation == MutationType.BENEFICIAL:
            fuzzedhalf += fuzzamt  # beneficial inheritance slightly more than half

        # uncomment below if want to see info about mutations...
        '''
        if mutation != MutationType.NO_MUTATION:
            mut_type = 'DEL' if mutation == MutationType.DELETERIOUS else 'BEN'
            print(f'@ t={current_time} {new_symbiont._id} {mut_type} surplus: {half} {fuzzedhalf}')
        '''

        #print(f">>>>>>>>> FUZZED INHERIT SURPLUS = {new_symbiont._photosynthate_surplus}")

        new_symbiont._photosynthate_surplus = fuzzedhalf     # child gets (fuzzed) half
        self._photosynthate_surplus -= fuzzedhalf            # parent decremented by half

        new_symbiont._surplus_on_arrival = new_symbiont._photosynthate_surplus

        #print(">>>>>>>>> HALVING WORKS?= ", \
        #    (old == new_symbiont._photosynthate_surplus + self._photosynthate_surplus))

        # this divided cell may not have a cell to reside in or may be infecting
        # outside our environment, and if so, there is no need to compute a
        # production rate or initial events...
        # (calling the methods below will actually cause a problem)
        if new_symbiont._cell is not None:
            assert(new_symbiont._production_rate == self._production_rate)

            new_symbiont._production_rate = \
                new_symbiont._computeProductionRate(is_copy = True, current_time = current_time)  # fuzzing inside!

            new_symbiont._scheduleInitialEvents(current_time)
            new_symbiont._setNextEvent()

            # 5 Oct 2016
            new_symbiont._cells_inhabited         = [str(new_symbiont._cell.getRowCol()).replace(', ',',')]
            new_symbiont._inhabit_times           = [current_time]
            new_symbiont._hcds_of_cells_inhabited = [new_symbiont._cell.getDemand()]

        return new_symbiont # return the newly created copy

    ##############################################################################################
    def _computeSurplusAtEventEnd(self, this_time: float, next_time: float, state: SymbiontState) \
                -> list[float, float or None, float or None]:
        ''' This method computes the amount of photosynthate surplus that will
            be present at the end of the next event, indicating whether the
            symbiont will make it successfully to the event or whether we
            should schedule an exit strategy for the symbiont.  
        Parameters:
            this_time: this symbiont's current event time (float)
            next_time: this symbiont's next event time (float)
            state: the current state of the symbiont (e.g., IN_G0)
        Returns:
            a list containing:
                (a) the computed surplus at the end of the event
                (b) the computed digestion time (or None, if not being digested)
                (c) an exit expulsion time (or None, if not exiting)
        '''

        # if this symbiont can't produce photosynthate at a rate sufficient
        # to meet the host cell's demand through the next event, will need
        # to eventually (elsewhere) schedule the exit strategy...
        time_diff = next_time - this_time
        produced = time_diff * self._production_rate
        demanded = time_diff * self._cell.getDemand()
        expended = 0

        # only during mitosis (coming out of G1SG2M) should we compute and
        # expended photosynthate as a cost for undergoing mitosis (state will
        # be IN_G1SG2M)
        if state == SymbiontState.IN_G1SG2M:
            expended = time_diff * self._mitotic_cost_rate

        surplus_at_end = self._photosynthate_surplus + produced - demanded - expended

        '''
        logging.debug(f'\t>>>time of next event: {next_time}')
        logging.debug(f'\t>>>produced:           {produced}')
        logging.debug(f'\t>>>demanded:           {demanded}')
        logging.debug(f'\t>>>expended:           {expended}')
        logging.debug(f'\t>>>surplus then:       {surplus_at_end}')
        '''

        t_d  = None # time of digestion, if any computed below
        t_ee = None # time of exit expulsion, if any computed below

        if surplus_at_end < 0:
            # the symbiont will not survive through the next event in this cell;
            # so, determine when the surplus will drop below 0 and then
            # schedule a digestion event or, if lucky with the coin flip, an
            # exit-expulsion; consider endpoints of line (t_c,s_c) and (t_e,s_e)
            # where 'c' is current, 'e' is end, and s_e < 0;  then slope of line
            # is
            #         m = (s_e - s_c) / (t_e - t_c)
            # and the computed time of digestion t_d can be computed by solving
            # y - y1 = m(x - x1) using (t_c,s_c) and solving for x when y = 0:
            #         t_d = t_c - (s_c/m)
            t_c = this_time
            s_c = self._photosynthate_surplus
            t_e = next_time
            s_e = surplus_at_end
            assert((t_e - t_c) > 0.0)
            m   = (s_e - s_c) / float(t_e - t_c)
            t_d = t_c - (s_c / m)  # computed time of digestion
            #logging.debug(f'\t>>>time of digestion:  {t_d}')

            # determine if this to-be-digested symbiont is lucky enough to
            # instead exit first; use the correct stream and probability...
            stream_prob = None
            prob        = None
            stream_exit = None
            if state == SymbiontState.IN_G0:
                stream_prob = Stream.DIGESTION_VS_ESCAPE_G0
                prob        = self._my_clade.getG0EscapeProb()
                stream_exit = Stream.TIME_G0_ESCAPE
            elif state == SymbiontState.IN_G1SG2M:
                stream_prob = Stream.DIGESTION_VS_ESCAPE_G1SG2M
                prob        = self._my_clade.getG1SG2MEscapeProb()
                stream_exit = Stream.TIME_G1SG2M_ESCAPE
            else:
                assert(False) # should never get here if state is not one of the above

            p = RNG.uniform(0, 1, stream_prob)
            if p < prob:
                # lucky -- will have an exit expulsion before digesting
                t_ee = RNG.uniform(this_time, t_d, stream_exit)

        if t_d is not None or t_ee is not None:
            assert(surplus_at_end < 0) # sanity check

        return [surplus_at_end, t_d, t_ee]

    #############################################################################
    def _computeNextEndOfG0(self, current_time: float) -> float:
        ''' Method to compute the next time of an end-of-G0 event, using
                current_time + avg G0 length +/- small fudge
        Parameters:
            current_time: the current event (simulation) time (float)
        Returns:
            the next end-of-G0 time for this symbiont (float)
        '''
        # using normal distribution
        m = self._my_clade.getG0Length()
        f = self._my_clade.getG0Fuzz()
        g0_time = RNG.fuzz(m, f, Stream.END_G0)  # fuzzed version of G0 length
        #print(f">>> G0: {g0_time}")

        next_time = current_time + g0_time
        self._g0_times.append(g0_time)
        return next_time

    #############################################################################
    def _computeNextEndOfG1SG2M(self, current_time: float) -> float:
        ''' Method to compute the next time of an end-of-G1SG2m event, using
                current_time + avg G1SG2M length +/- small fudge
        Parameters:
            current_time: the current event (simulation) time (float)
        Returns:
            the next end-of-G1SG2M time for this symbiont (float)
        '''
        # Using normal distribution
        m = self._my_clade.getG1SG2MLength()
        f = self._my_clade.getG1SG2MFuzz()
        g1sg2m_time = RNG.fuzz(m, f, Stream.END_G1SG2M)  # fuzzed version of G1SG2M length
        #print(f">>> G1SG2M: {g1sg2m_time}")

        next_time = current_time + g1sg2m_time
        self._g1sg2m_times.append(g1sg2m_time)
        return next_time

    #############################################################################
    def endOfG0(self, current_time: float) -> None:
        ''' Method to handle the transition from end of G0 into G1SG2M.  This
            method will compute the amount of photosynthate surplus the symbiont
            will have before the end of the coming G1SG2M event. If all goes 
            well for the symbiont, the surplus will be positive and an end-of-
            G1SG2M event can be scheduled.  If the surplus is determined to be
            negative by the time of the future end-of-G1SG2M event, the symbiont
            will either been digested or will have escaped prior -- so need to
            schedule that earlier event instead.
        Parameters:
            current_time: the current simulation time -- @ end of G0 (float)
        '''

        self._time_of_next_end_g0 = INFINITY
        assert(self._prev_event_type == EventType.ARRIVAL or \
               self._prev_event_type == EventType.END_G1SG2M)

        # first, compute the photosynthate surplus since last event (the last
        # event will have been an end-of-G1/S/G2/M); 
        # computed surplus should never be negative here so pass None as state
        # (_computeSurplusAtEventEnd uses state when determining digestion or exit)
        [surplus_at_end, time_of_digestion, time_of_exit] = \
            self._computeSurplusAtEventEnd(self._prev_event_time, current_time, \
                                          state = SymbiontState.IN_G0)

        assert(time_of_digestion == None) # sanity check -- should not have a value
        assert(time_of_exit == None)      # sanity check -- should not have a value

        self._photosynthate_surplus = surplus_at_end

        # now, compute the amount of photosynthate produced, demanded by the
        # host cell, and expended on mitosis during the entire G1/S/G2/M period
        time_of_end_g1sg2m = self._computeNextEndOfG1SG2M(current_time)
        [surplus_at_end, time_of_digestion, time_of_exit] = \
            self._computeSurplusAtEventEnd(current_time, time_of_end_g1sg2m, \
                                          SymbiontState.IN_G1SG2M)

        if surplus_at_end < 0:
            # the mitosis will not complete successfully in this cell environment; 
            # so, determine when the surplus will drop below 0 and then schedule a 
            # digestion event or, if lucky with the coin flip, an exit-expulsion as 
            # the mitosis event is progressing...
            self._time_of_digestion = time_of_digestion
            if time_of_exit is not None:
                self._time_of_escape = time_of_exit # from _computeSurplusAtEventEnd()
        else:
            # the mitosis will complete successfully, so schedule it;
            # we could be a little more computationally efficient and update
            # the photosynthate here for the end of G1SG2M event rather than 
            # recompute there -- but it makes debugging harder...
            self._time_of_next_end_g1sg2m = time_of_end_g1sg2m

        # whatever the event might be, set the next event for this symbiont
        self._prev_event_time = current_time
        self._prev_event_type = EventType.END_G0
        self._setNextEvent()

    #############################################################################
    def endOfG1SG2M(self, current_time: float) -> list[SymbiontState, 'Symbiont']:
        ''' Method to handle the transition from end of G1SG2M, when the mitosis
            is completing, into the next G0 state.  This method checks for all
            possible scenarios of what can happen when division is successful:
                - both parent and child find happy cell homes
                    - parent may stay and child may move into new cell
                    - child may stay and parent may move into new cell
                    - note -- could be that an open cell is outside the scope
                      of our 2D grid, if the parent lives on the top or bottom
                      border
                - no room adjacent, so one of the parent or child is evicted
            This will also update the next end-of-G0 time for the parent,
            should that be able to occur (or scheduling digestion or eviction
            prior should the parent not be able to make it to the next
            end-of-G0).
        Parameters:
            current_time: the current simulation time -- @ end of G1SG2M (float)
        Returns:
            a list containing the status resulting from the mitosis, and the
            resulting child symbiont; possible statuses returned:
                SymbiontState.PARENT_INFECTS_OUTSIDE
                SymbiontState.CHILD_INFECTS_OUTSIDE
                SymbiontState.BOTH_STAY
                SymbiontState.PARENT_NO_AFFINITY (division affinity)
                SymbiontState.CHILD_NO_AFFINITY (division affinity)
                SymbiontState.PARENT_EVICTED
                SymbiontState.CHILD_EVICTED
        '''

        self._time_of_next_end_g1sg2m = INFINITY
        return_status_and_child = None
    
        assert(self._prev_event_type == EventType.END_G0)
    
        # remember that, to save computation, we could have already computed the 
        # photosynthate surplus to this point (see comment above), but it makes
        # debugging easier if this is here;
        # computed surplus should never be negative here so pass None as state
        # (_computeSurplusAtEventEnd uses state when determining digestion or exit);
        #
        # also note that in endOfG0(), we precomputed what the additional cost of
        # division would entail, and if more than symbiont can afford, would never
        # have gotten here, but would have escaped or been digested earlier
        #
        [surplus_at_end, time_of_digestion, time_of_exit] = \
            self._computeSurplusAtEventEnd(self._prev_event_time, current_time, \
                                          state = SymbiontState.IN_G1SG2M)
    
        assert(time_of_digestion == None) # should not get here otherwise...
        assert(time_of_exit == None)      # should not get here otherwise...
    
        self._photosynthate_surplus = surplus_at_end
        self._num_divisions = self._num_divisions + 1
    
        # try to find an open cell for the new symbiont (note: it may be
        # a (modeled) cell outside our environment if the original symbiont is
        # in the top or bottom row of the host cell grid)
        open_cell = self._checkForOpenAdjacentCell()
    
        # self._cell is the parent's cell
        # inside _SymbiontCopy, self refers to parent and new_symbiont to child
        # To do:
        # - if there is an open cell (either inside or outside), flip fair coin
        #       to determine who stays and who goes
        #   - cases:
        #       (a) open cell outside environment:
        #           child goes (no Cell to SymCopy), parent stays (no change to self._cell)
        #           child stays (self._cell to SymCopy), parent goes (self._cell = no Cell)
        #       (b) open cell inside the environment:
        #           child goes (open_cell to SymCopy), parent stays (no change to self._cell)
        #           child stays (self._cell to SymCopy), parent goes (self._cell = open_cell)
        # - if no open cell, flip fair coin to determine who stays
        #       (already modeled below -- use as an example)
        #
        # - for event scheduling:
        #       - update parent's next time of endG0 only if parent is not evicted
        #             or if parent is not moving to cell outside our grid
    
        #########################################################################
        if open_cell == SymbiontState.CELL_OUTSIDE_ENVIRONMENT:
        #########################################################################
            # this is not an eviction -- we are presuming the new symbiont is
            # infecting a cell outside the scope of our modeled environment
            # (above top row or below bottom row); 
            # child is created but it or parent presumed gone outside our
            # environment -- call the _SymbiontCopy method
            prob = RNG.uniform(0, 1, Stream.EVICTION)
            if prob < self._my_clade.getParentEvictionProb():
                ######################################################
                ## child stays in current cell, parent infects outside
                ######################################################
                #print(f"Parent infecting cell along border {self._id}")
                child = self._SymbiontCopy(self._cell, current_time)
                self._cell.setSymbiont(child, current_time) # update cell to contain child
                self._cell = None  # parent infects outside (i.e., no cell in model)
                return_status_and_child = [SymbiontState.PARENT_INFECTS_OUTSIDE, child]
                # info on current cell inhabited by child occurs in _SymbiontCopy
            else:
                ######################################################
                ## parent stays in current cell, child infects outside
                ######################################################
                #print(f"Child infecting cell along border {self._id}")
                no_cell = None
                child = self._SymbiontCopy(no_cell, current_time) 
                return_status_and_child = [SymbiontState.CHILD_INFECTS_OUTSIDE, child]
                # no need to update new cells inhabited for either parent or child
            #
        #########################################################################
        elif open_cell is not None:  # there is an open cell for child or parent
        #########################################################################
            prob = RNG.uniform(0, 1, Stream.EVICTION)
            if prob < self._my_clade.getParentEvictionProb():
                ## child stays in current cell, parent moves to the new open cell
                #print(f"Parent goes to open cell {self._id}")
                child = self._SymbiontCopy(self._cell, current_time)
                self._cell.setSymbiont(child, current_time) # current cell now contains child
    
                # use affinity values to determine if symbiont is phagocytosed
                phagocytosed = Symbiont._determinePhagocytosis(self._my_clade, is_arrival = False)
                if phagocytosed:
                    self._cell = open_cell
                    open_cell.setSymbiont(self, current_time)  # open cell now contains parent
                    # info on new cell inhabited by child occurs in _SymbiontCopy;
                    # but need to record info on parent switching to new open cell
                    self._cells_inhabited.append(str(open_cell.getRowCol()).replace(', ',','))
                    self._inhabit_times.append(current_time)
                    self._hcds_of_cells_inhabited.append(open_cell.getDemand())
                    return_status_and_child = [SymbiontState.BOTH_STAY, child]
                else:
                    # similar to parent evicted
                    self._cell = None # parent now homeless
                    return_status_and_child = [SymbiontState.PARENT_NO_AFFINITY, child]
            else:
                ## parent stays in current cell, child moves to the new open cell
                ## call symbiont copy constructor, place into open cell
                #print(f"Child goes to open cell {child.id}")
            
                # use affinity values to determine if symbiont is phagocytosed
                phagocytosed = Symbiont._determinePhagocytosis(self._my_clade, is_arrival = False)
                if phagocytosed:
                    # info on new cell inhabited by child occurs in _SymbiontCopy
                    child = self._SymbiontCopy(open_cell, current_time)
                    open_cell.setSymbiont(child, current_time)
                    return_status_and_child = [SymbiontState.BOTH_STAY, child]
                    # no need to update new cells inhabited for either parent or child
                else:
                    no_cell = None
                    child = self._SymbiontCopy(no_cell, current_time)
                    return_status_and_child = [SymbiontState.CHILD_NO_AFFINITY, child]
            # 
        #########################################################################
        else: # there is no open cell for a new symbiont
        #########################################################################
            # there is no room for an additional symbiont, so flip a coin; 
            # if 0, the child would have been created but ejected into the pool;
            #     still, call _SymbiontCopy to make sure photosynthate is evenly
            #     divided -- parents stays put;
            # if 1, the parent will be evicted into the pool, and the copied 
            #     child will go into the current cell
            prob = RNG.uniform(0, 1, Stream.EVICTION) 
            if prob < self._my_clade.getParentEvictionProb():
                ## child stays in current cell, parent evicted into pool;
                ## call symbiont copy constructor, place child into current cell
                #print(f"Parent evicted into pool {self._id}")
                child = self._SymbiontCopy(self._cell, current_time)
                self._cell.setSymbiont(child, current_time)
                self._cell = None  # parent now homeless
                return_status_and_child = [SymbiontState.PARENT_EVICTED, child]
                # info on new cell inhabited by child occurs in _SymbiontCopy
            else:
                ## parent stays in current cell, child evicted into pool;
                # child is created but presumed gone into the pool, so no cell
                #print(f"Child evicted into pool {child.id}")
                no_cell = None
                child = self._SymbiontCopy(no_cell, current_time)
                return_status_and_child = [SymbiontState.CHILD_EVICTED, child]
                # no need to update new cells inhabited for either parent or child
        #########################################################################
    
        # only in the parent-evicted or parent-infects-outside cases above 
        # do we NOT try to update the parent's next end of G0
        if not (return_status_and_child[0] == SymbiontState.PARENT_EVICTED or \
                return_status_and_child[0] == SymbiontState.PARENT_INFECTS_OUTSIDE or \
                return_status_and_child[0] == SymbiontState.PARENT_NO_AFFINITY):  
            # must make sure that parent can make it through this next G0 event
            # (e.g., could be producing at rate less than host cell demand, but
            # banked photosynthate is sufficient to allow it through a few events)
            time_of_end_of_g0 = self._computeNextEndOfG0(current_time)
            # note the parent's photosynthate has already been divided in _SymbiontCopy
            [surplus_at_end, time_of_digestion, time_of_exit] = \
                self._computeSurplusAtEventEnd(current_time, time_of_end_of_g0, \
                                              SymbiontState.IN_G0)
          
            if surplus_at_end < 0:
                self._time_of_digestion = time_of_digestion
                if time_of_exit is not None:
                    self._time_of_escape = time_of_exit
            else:
                self._time_of_next_end_g0 = time_of_end_of_g0
    
        # whatever the event might be, set the next event to occur for this symbiont
        self._prev_event_time = current_time
        self._prev_event_type = EventType.END_G1SG2M
        self._setNextEvent()

        return return_status_and_child

    #############################################################################
    def _checkForOpenAdjacentCell(self) -> Cell or None or SymbiontState:
        ''' method to check for an open adjacent cell on successful mitosis
        Returns:
            a Cell object that can be used for hosting the result of mitosis,
            or None if no open cell, or SymbiontState.CELL_OUTSIDE_ENVIRONMENT
            if the (modeled) open cell is determined to be outside the scope
            of our 2D grid of host cells (e.g., above the top row or below the
            bottom row)
        '''

        # check the Moore neighborhood at random
        row, col  = self._cell.getRowCol()
        positions = [(-1,-1),(-1,0),(-1,1), \
                     ( 0,-1),       ( 0,1), \
                     ( 1,-1),( 1,0),( 1,1)]
    
        num_open_inside_environment = 0 # will be used for border cases
    
        RNG.shuffle(positions, Stream.CHECK_FOR_OPEN_CELL)
    
        open_cell = None
        found     = False
        while (not found) and len(positions) > 0:
            pos = positions.pop()  # grab the first position from the random shuffle
    
            # remember that the sponge canal wraps horizontally (cols) but not 
            # vertically (rows) since we are modeling only a slice of the canal
            row_offset = pos[0]
            if row + row_offset < 0 or row + row_offset >= Parameters.NUM_ROWS:
                continue
    
            col_offset = pos[1]
            candidate_cell = Symbiont.sponge.getCell( row + row_offset, \
                (col + col_offset) % Parameters.NUM_COLS )  # wrap the column
            if not candidate_cell.isOccupied():
                open_cell = candidate_cell
                found = True
    
        # if there is an open cell and the parent lives on the top or bottom
        # row, the occupied cell will be in the Moore neighborhood within our
        # 2D grid with probability 5/8 (e.g., on top row, cells W,E,SW,S,SE are 
        # inside our modeled grid but cells NW,N,NE are outside our grid)
        if open_cell is not None and (row == 0 or row == Parameters.NUM_ROWS - 1):
            p = RNG.uniform(0, 1, Stream.INFECT_CELL_OUTSIDE)
            # with prob 3/8, infect outside; with prob 5/8, infect inside
            if p < 0.375:  # probability 3/8 infects outside
                open_cell = SymbiontState.CELL_OUTSIDE_ENVIRONMENT
    
        return open_cell

    #############################################################################
    def digestion(self, current_time: float) -> None:
        ''' method to handle digestion of the current symbiont, simply
            removing the symbiont from its current host cell
        Parameters:
            current_time: current simulation time -- time of digestion (float)
        '''
        # 8 Oct 2016: keep track of photosynthate at end, even if leaving...
        # useful for the detailed csv per-symbiont tracking
        self._photosynthate_surplus = 0  # digested, so must be 0!
        self._cell.removeSymbiont(current_time)

    #############################################################################
    def escape(self, current_time: float) -> None:
        ''' method to handle the current symbiont escaping digestion, simply
            removing the symbiont from its current host cell (presumed to be
            going back to the pool)
        Parameters:
            current_time: current simulation time -- time of escape (float)
        '''
        # 8 Oct 2016: keep track of photosynthate at end, even if leaving...
        # useful for the detailed csv per-symbiont tracking
        self._photosynthate_surplus = 0  # escaping, so must be 0!
        self._cell.removeSymbiont(current_time)

    #############################################################################
    def denouement(self, current_time: float) -> None:
        ''' method to handle the current symbiont's denouement and transition 
            back to the pool, simply removing the symbiont from its current
            host cell (presumed to be going back to the pool) and appropriately
            computing the symbiont's photosynthate surplus at denouement (for
            tracking)
        Parameters:
            current_time: current simulation time -- time of denouement (float)
        '''
        # 8 Oct 2016: keep track of photosynthate at end, even if leaving...
        # useful for the detailed csv per-symbiont tracking
        # (see comment in endOfG0 on use of computeSurplus method...)
        [surplus_at_end, time_of_digestion, time_of_exit] = \
            self._computeSurplusAtEventEnd(self._prev_event_time, current_time, \
                                          state = None)
        assert(time_of_digestion == None)  # sanity check
        assert(time_of_exit == None)       # sanity check
        assert(surplus_at_end >= 0)  # o/w, shouldn't have made it to denouement

        self._photosynthate_surplus = surplus_at_end
        self._cell.removeSymbiont(current_time)

    #############################################################################
    @staticmethod
    def _determinePhagocytosis(clade: Clade, is_arrival: bool) -> bool:
        ''' static method to determine whether phagocytosis happens based
            on clade-specific probability
        Parameters:
            clade: Clade object corresponding to a symbiont clade
            is_arrival: True if symbiont is arriving, False o/w
        Returns:
            True if the symbiont will be phagocytosed (whether on arrival
            or division), False o/w
        '''
        #############################################################################
        # New method, as of spring 2019: each algal clade takes on a different 
        # arrival and division affinity "strategy", denoted by a number, which has
        # a different probability
        #    u <- fair coin flip using either 
        #           Stream.ARRIVAL_AFFINITY or Stream.DIVISION_AFFINITY
        #    p <- clade-specific probability of entry/maintenance based
        #           (provided by user in input file)
        #    if (u < p) {phagocytosed}  else  {not phagocytosed}
        #############################################################################
        if is_arrival:
            affinity_prob = RNG.uniform(0, 1, Stream.ARRIVAL_AFFINITY)  # flip a coin
            # retrieve arrival affinity probability based on clade
            clade_prob = clade.getArrivalAffinityProb()
        else:
            affinity_prob = RNG.uniform(0, 1, Stream.DIVISION_AFFINITY) # flip a coin
            # retrieve division affinity probability based on clade
            clade_prob = clade.getDivisionAffinityProb()

        # if u < p, return True (phagocytosed), o/w False (not phagocytosed)
        return affinity_prob < clade_prob

    #############################################################################
    # def eviction(self, current_time: float) -> None:
        # note there is no separate eviction event because we handle the eviction
        # within endOfG1SG2M() -- to compute stats appropriately, in the case of
        # parent eviction we need a reference to the cell, so we don't handle this
        # as a separate event

    #############################################################################
    def getNextEvent(self) -> tuple[float, EventType]:
        ''' method to return the next event time and type for this symbiont,
            as stored internally in the symbiont
        Returns:
            a tuple containing the time of the next event and the event type
        '''
        return (self._next_event_time, self._next_event_type)

    #############################################################################
    def _setNextEvent(self) -> None:
        ''' method to update the symbiont's internal state keeping track of its
            next event time and type to occur
        '''
        # NOTE: the order of occurrence is important here -- an event higher
        # in the order takes precedence of an event lower in the order should
        # there be identical event times
        self._next_event_time = self._time_of_next_end_g0
        self._next_event_type = EventType.END_G0

        if self._time_of_next_end_g1sg2m < self._next_event_time:
            self._next_event_time = self._time_of_next_end_g1sg2m
            self._next_event_type = EventType.END_G1SG2M

        if self._time_of_escape < self._next_event_time:
            self._next_event_time = self._time_of_escape
            self._next_event_type = EventType.ESCAPE

        if self._time_of_digestion < self._next_event_time:
            self._next_event_time = self._time_of_digestion
            self._next_event_type = EventType.DIGESTION

        if self._time_of_denouement < self._next_event_time:
            self._next_event_time = self._time_of_denouement
            self._next_event_type = EventType.DENOUEMENT

    #############################################################################
    def _scheduleInitialEvents(self, current_time: float) -> None:
        # set up the long-term denouement time (which we presume is because the
        # symbiont leaves the host cell in order to reproduce) -- this may be
        # superseded by the exit strategy computation below...

        m = self._my_clade.getAvgResidenceTime()
        f = self._my_clade.getResidenceFuzz()
        residence_time = RNG.fuzz(m, f, Stream.TIME_DENOUEMENT)
        self._time_of_denouement = current_time + residence_time
        #print(f">>> RESTIME: {residence_time}")

        # set up the next mitosis to be on schedule, +/- fudge factor;
        # this allows us to model "accumulated bad decisions" RE division;
        # mitosis is scheduled using two events (see 17 Mar 2015 email with
        # photo of board notes):
        #     1) end of G0 / start of G1,S,G2,M
        #     2) end of M / start of G0
        # in this way, if the dividing cell's photosynthate surplus goes
        # negative at any time during the G1,S,G2,M phase -- when the additional
        # subtraction due to cost of mitosis occurs -- that cell will either
        # be digested or exit-expulsed, and the division never occurs in our 
        # model (it could be the case that if exit-expulsed, the cell would 
        # divide elsewhere, but that is beyond our model scope);
        # 
        # moreover, with respect to the times for the G0,G1,S,G2,M process, we
        # presume that _only_ G0 will vary for hi vs med vs low reproducers;
        # the average time for G1,S,G2,M will be the same regardless -- in other
        # words low reproducers will spend more time in G0 banking photosynthate
        # before entering the "going to divide" state of G1,S,G2,M from which there
        # is no turning back once committed
        self._time_of_next_end_g0 = self._computeNextEndOfG0(current_time)

        # if this symbiont can't produce photosynthate at a rate sufficient
        # to meet the host cell's demand through G0, schedule the exit strategy...
        [surplus_at_end, time_of_digestion, time_of_exit] = \
            self._computeSurplusAtEventEnd(current_time, self._time_of_next_end_g0, \
                                          SymbiontState.IN_G0)

        if surplus_at_end < 0:
            # the symbiont will not survive through the first G0 in this cell;
            # so, schedule a digestion event or, if lucky with the coin flip, an 
            # exit-expulsion 
            self._time_of_digestion = time_of_digestion
            if time_of_exit is not None:
                # rapid expulsion, uniformly distributed between now 
                # and when the digestion would naturally occur
                self._time_of_escape = time_of_exit

    #############################################################################
    def _computeProductionRate(self, is_copy: bool, current_time: float) -> float:
        ''' computes the photosynthetic production rate (PPR) of this symbiont
        Parameters:
            is_copy: True if this symbiont is arriving via mitosis; False o/w
            current_time: current simulation time (float)
        '''
        # photosynthetic production rate is a function of the clade (corresponding
        # rates defined in Parameters) and host cell location, with production rate
        # decreasing linearly from north to south (moving away from the sun);

        # for clade X:
        # for now presume that rho_X decreases linearly from full rho_X at the
        # topmost row to rho_X / k at the lowest row; hence, the endpoints of the
        # corresponding line are (rho_X, 0) and (rho_X / k, N-1), where N is the
        # maximum number of rows;  after dervation, this should give a line
        # equation of 
        #           y = rho_X + ((1-k)/k)*(x*rho_X/(N-1))
        # where y is the production rate and x is the corresponding row
        k = self._my_clade.getPhotosyntheticReduction()
        #######################################################################
        if is_copy:
            rho = self._production_rate  # this is an exact copy of parent
        else:
            rho = self._my_clade.getPPR()
        #######################################################################
        num_rows, num_cols = Symbiont.sponge.getDimensions()
        row, col = self._cell.getRowCol() # row x in the equation above
        rate = rho + (float(1-k)/k) * (row*rho/float(num_rows-1))

        ## 12 Apr 2016
        # for fuzzing, use normal with  95% of the data b/w (mu +/- mu*f) -- 
        # see implementation in rng.py; here there can be different fuzzing
        # parameters depending on whether this is an "original" symbiont or a
        # copy via division
        if is_copy:
            # use multi-distro division fuzzing -- see implementation in rng.py
            [fuzz_amt, mutation] = RNG.divfuzz(rate, self._my_clade, Stream.PHOTOPROD_MUTATION)
            fuzzed_rate = rate
            if mutation == MutationType.DELETERIOUS:
                fuzzed_rate -= fuzz_amt  # deleterious photoprod reduces
            else: # mutation == MutationType.BENEFICIAL:
                fuzzed_rate += fuzz_amt  # beneficial photoprod increases
            # uncomment below if want to see info about mutations...
            '''
            if mutation != MutationType.NO_MUTATION:
                mut_type = 'DEL' if mutation == MutationType.DELETERIOUS else 'BEN'
                print(f'@ t={current_time} {self._id} {mut_type} ppr: {rate} {fuzzed_rate}")
            '''
        else:
            # use normal -- see implementation in rng.py
            m = rate
            f = self._my_clade.getPPRFuzz()
            fuzzed_rate = RNG.fuzz(m, f, Stream.PHOTOPROD)

        return fuzzed_rate  # y in the equation above

    #############################################################################
    ''' simple getter/accessor methods '''
    def getID(self)            -> int:       return self._id
    def getCladeNumber(self)   -> int:       return self._clade_number
    def getArrivalTime(self)   -> float:     return self._arrival_time
    def getPrevEventType(self) -> EventType: return self._prev_event_type

    #############################################################################
    def __str__(self) -> str:
        ''' create a useful string for printing a symbiont
        Returns:
            str representation of this Symbiont object
        '''
        row, col = self._cell.getRowCol() if self._cell is not None else [-1,-1]
        string = f'Symbiont {self._id}' \
            + f': @({row},{col})\n\t' \
            + f'clade        : {self._clade_number}\n\t' \
            + f'cell         : {self._cell}\n\t' \
            + f'prod rate    : {self._production_rate}\n\t' \
            + f'surplus      : {self._photosynthate_surplus}\n\n\t' \
            + f'next event time  : {self._next_event_time}\n\t' \
            + f'next event type  : {self._next_event_type.name}\n\t' \
            + f'arrival time     : {self._arrival_time}\n\t' \
            + f'last event time  : {self._prev_event_time}\n\t' \
            + f'time of end-G0   : {self._time_of_next_end_g0}\n\t' \
            + f'time of end-G1->M: {self._time_of_next_end_g1sg2m}\n\t' \
            + f'time of exit exp : {self._time_of_escape}\n\t' \
            + f'time of digest   : {self._time_of_digestion}\n\t' \
            + f'time of res exp  : {self._time_of_denouement}'
        return string

    ###################################################################################
    def csvOutputOnExit(self, current_time: float, exit_status: SymbiontState) -> None:
        ''' method to write per-symbiont information to CSV file (if requested)
        Parameters:
            current_time: current simulation time (float)
            exit_status: exit status of symbiont (one from SymbiontState enumeration)
        '''
        # Note there are four ways for a symbiont to exit the system:
        #   (1) residence expulsion -- just ends its natural time in the sponge
        #   (2) digested (during mitosis or not)
        #   (3) escape (during mitosis or not)
        #   (4) evicted (could be the parent or the child)
        # This method should be called whenever one of those happens, giving CSV
        # output for the statistics of that exiting symbiont.
        #
        if not Symbiont._write_csv: return
        strval  = str(self._id)                + ','    # overall symbiont id number
        strval += str(self._how_arrived.name)  + ','    # via pool or division
        strval += str(self._parent_id)         + ','    # id of parent, -1 via pool
        strval += str(self._agent_zero)        + ','    # id of ultimate ancestor
        strval += str(self._clade_number)      + ','
        strval += str(self._mitotic_cost_rate) + ','
        strval += str(self._production_rate)   + ','
        strval += str(self._arrival_time)      + ','    # arrival time
        strval += str(current_time)            + ','    # exit time (1 of 4 above)
        strval += str(exit_status.name)        + ','
        ## begin added 31 Oct 2016
        strval += str(self._prev_event_time)             + ','
        strval += str(self._prev_event_type.name)        + ',' 
        ## end added 31 Oct 2016
        strval += str(current_time - self._arrival_time) + ','  # residence time
        strval += str(self._surplus_on_arrival)          + ','  # surplus @ arrival
        strval += str(self._photosynthate_surplus)       + ','  # surplus @ exit
        strval += str(self._num_divisions)               + ','  # num successful divs
        strval += str(self._time_of_escape)              + ','
        strval += str(self._time_of_digestion)           + ','
        strval += str(self._time_of_denouement)          + ','
        if exit_status == SymbiontState.STILL_IN_RESIDENCE:
            strval += SymbiontState.STILL_IN_RESIDENCE.name + ','
        else:
            strval += "NOT_IN_RESIDENCE,"
        # append cells (perhaps multiple) inhabited by symbiont -- separate w/ ;
        for i in range(len(self._cells_inhabited)):
            strval += ('"' if i == 0 else ';') + str(self._cells_inhabited[i])
        if len(self._cells_inhabited) > 0: strval += '"'
        # append times (perhaps multiple) cells were inhabited by symbiont
        strval += ','
        for i in range(len(self._inhabit_times)):
            strval += ('' if i == 0 else ';') + str(self._inhabit_times[i])
        # append hcds (perhaps multiple) of cells inhabited by symbiont
        strval += ','
        for i in range(len(self._hcds_of_cells_inhabited)):
            strval += ('' if i == 0 else ';') + str(self._hcds_of_cells_inhabited[i])
        # append g0 times symbiont experienced; separate diff times by semicolon
        strval += ','
        # if symbiont is a child immediately evicted or child who infected outside
        # that means it received a g0 time that was never used -- let's not
        # include the g0 time in the output; a parent who finished g1sg2m but was 
        # evicted or infected outside does not get assigned a new g0 time -- see 
        # endG1SG2M(); for a symbiont who is digested or escapes during G0 (added 
        # via _computeNextEndOfG0() near end of endG1SG2M()), want to keep that g0
        # time in the output as it was "partially" used...  (see overall comments
        # appended to the bottom of this program)
        len_g0_times = len(self._g0_times)
        if exit_status == SymbiontState.CHILD_INFECTS_OUTSIDE or \
           exit_status == SymbiontState.CHILD_EVICTED: len_g0_times -= 1
        for i in range(len_g0_times):
            strval += ('' if i == 0 else ';') + str(self._g0_times[i])
        #
        # append g1sg2m times symbiont experienced; separate diff times by semicolon
        strval += ','
        for i in range(len(self._g1sg2m_times)):
            strval += ('' if i == 0 else ';') + str(self._g1sg2m_times[i])
        #
        # append cnt of open cells (at division) seen; separate diff times by semicolon
        #strval += ','
        #for i in range(len(self._cells_at_division)):
        #    strval += ('' if i == 0 else ';') + str(self._cells_at_division[i])
        #
        strval += '\n'
        Symbiont._csv_file.write(strval)
        Symbiont._csv_writes += 1

    #############################################################################
    @classmethod
    def computeCumulativeCladeProportions(cls) -> None:
        ''' class-level method to sets up an array of cumulative proportions
            (probabilities) so that different clades can have different
            probabilities of arriving -- used when generating a symbiont
            arrival
        '''
        cls.clade_cumulative_proportions = numpy.cumsum(Parameters.CLADE_PROPORTIONS)
        # set last entry to 1.0 just to be safe (avoid roundoff errors)
        cls.clade_cumulative_proportions[-1] = 1.0

    @classmethod
    def openCSVFile(cls, csv_fname: str) -> None:
        cls._write_csv = True
        cls._csv_file = open(csv_fname, "w")
        cls._csv_file.write(\
           'symbID,poolOrDiv,parent,agentZero,clade,mcr,ppr,'\
          +'arrTime,exitTime,exitStatus,lastEventTime,lastEventType,'\
          +'resTime,arrSurplus,exitSurplus,divs,'\
          +'tEsc,tDig,tRes,stillInRes,cells,inhabitTimes,'\
          +'hcds,g0Times,g1sg2mTimes,cellsAtDiv\n');

    @classmethod
    def csvOutputAtEnd(cls, current_time: float) -> None:
        ''' class-level method to write to CSV the per-symbiont information (if
            requested) at the end of the simulation
        Parameters:
            current_time: current simulation time @ end (float)
        '''
        if not cls._write_csv: return
        # write output for all those still in residence
        rows, cols = cls.sponge.getDimensions()
        for r in range(rows):
            for c in range(cols):
                cell = cls.sponge.getCell(r,c)
                symbiont = cell.getSymbiont()
                if symbiont is not None:
                    symbiont.csvOutputOnExit(current_time, SymbiontState.STILL_IN_RESIDENCE)
        Symbiont._csv_file.close()
  

    #############################################################################
    @classmethod
    def findOpenCell(cls) -> Cell:
        ''' static method to find and select an open cell at random among all
            available (unoccupied) cells in the entire Sponge grid
        Returns:
            the Cell object selected
        '''
        # create a list of all open cells
        open_cells = []
        for r in range(Parameters.NUM_ROWS):
            for c in range(Parameters.NUM_COLS):
                cell = cls.sponge.getCell(r,c)
                if not cell.isOccupied():
                    open_cells.append(cell)

        # we should never call this method unless there is at least one open cell
        assert(len(open_cells) > 0)  # sanity check

        # pick one @ random
        which = RNG.randint(0, len(open_cells)-1, Stream.OPEN_CELL_ON_ARRIVAL)
        return open_cells[which]

    #######################################################################################
    @classmethod
    def findOpenCellWithin(cls, min_row: int, max_row: int, min_col: int, max_col: int) -> Cell:
        ''' Static method to find and select an open cell at random among 
            available (unoccupied) cells within a particular section of the Sponge grid.
            Note that min_row and min_col are inclusive; max_row and max_col
            are exclusive.
        Paramters:
            min_row: the minimum row value to start the search (inclusive)
            max_row: the maximum row value to end the search (exclusive)
            min_col: the minimum col value to start the search (inclusive)
            max_col: the maximum col value to end the search (exclusive)
        Returns:
            the Cell object selected
        '''
        # create a list of all open cells
        open_cells = []
        for r in range(max_row - min_row):
            for c in range(max_col - min_col):
                cell = cls.sponge.getCell(min_row + r, min_col + c)
                if not cell.isOccupied():
                    open_cells.append(cell)

        # we should never call this method unless there is at least one open cell
        assert(len(open_cells) > 0)

        # pick one @ random
        which = RNG.randint(0, len(open_cells)-1, Stream.OPEN_CELL_ON_ARRIVAL)
        return open_cells[which]

    ################################################################################
    @classmethod
    def generateArrival(cls, current_time: float, num_symbionts: int) -> 'Symbiont or None':
        ''' method to generate a symbiont arrival
        Parameters:
            current_time: time of the arrival (current simulation time) (float)
            num_symbionts: total number of symbionts
        Returns:
            a new Symbiont object, if the sponge is not already full; None o/w
        '''
        # no need to even try if there are no available cells
        if num_symbionts == Parameters.NUM_ROWS * Parameters.NUM_COLS:
            #logging.debug('\tNo cells available')
            return None

        # now handle the arrival -- pick a clade at random using the previously
        # defined cumulative proportions for clades...
        prob = RNG.uniform(0, 1, Stream.CLADE)
        clade = 0
        while prob >= cls.clade_cumulative_proportions[clade]: clade += 1
    
        # now determine if there is appropriate affinity for infection;
        # first grab the clade object and use it to calculate arrival affinity
        this_clade = Clade.getClade(clade)

        # if this symbiont has insufficient arrival affinity for host, can't get in
        phagocytosed = cls._determinePhagocytosis(this_clade, is_arrival = True)
        if phagocytosed:
            # symbiont has arrival affinity - find an open cell for this symbiont
            open_cell = cls.findOpenCell()
            symbiont  = Symbiont(clade, open_cell, current_time)  
            open_cell.setSymbiont(symbiont, current_time)
        else:
            #logging.debug('\tNo affinity: clade %s' % (clade))
            symbiont = None

        return symbiont

######################################################################
# NOTES ON WHETHER LAST G0 TIME SHOULD BE EXCLUDED IN OUTPUT:
# (These notes are referenced in csvOutputOnExit() above)
# ####################################################################
# child infects outside
#     - does append new g0 in _scheduleInitialEvents
# child evicted
#     - does append new g0 in _scheduleInitialEvents
# parent evicted
#     - does not append new g0
# parent infects outside
#     - does not append new g0
# digestion during G0:     last event type is END_G1SG2M
#     - does append new g0 via _computeNextEndOfG0() near end of endG1SG2M()
# escape during G0:        last event type is END_G1SG2M
#     - does append new g0 via _computeNextEndOfG0() near end of endG1SG2M()
# digestion during G1SG2M: last event type is END_G0
#     - does not append a new g0 since doesn't get through G1SG2M
# escape during G1SG2M:    last event type is END_G0
#     - does not append a new g0 since doesn't get through G1SG2M
# denouement
#     - does not append new g0
