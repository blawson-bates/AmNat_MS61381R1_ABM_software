import bisect # http://pymotw.com/2/bisect/index.html
from enum import Enum

################################################################################
class EventType(Enum):
    # The event types are in a particular order below for ordering events;
    # so, for example, if two events have the same time, an escape event
    # takes precedence over a digestion event, etc.
    EVENT_MIN_SENTINEL  = -1
    ESCAPE              =  0
    DIGESTION           =  1
    END_G0              =  2  # when division process (G1/S/G2/M) starts
    END_G1SG2M          =  3  # when division actually occurs, & G0 re-starts
    DENOUEMENT          =  4
    ARRIVAL             =  5
    # EVICTION = ...  # no separate eviction event: see symbiont.py comments
    EVENT_MAX_SENTINEL  =  6

################################################################################
class Event:
    ''' Class to implement an event (to be stored in the corresponding event list)
        in the simulation model.  An event is determined by its time, the type
        of event, and the corresponding symbiont driving the event.
    '''
    __slots__ = ('_time', '_type', '_symbiont')

    #####################################
    def __init__(self, time: float, event_type: EventType, symbiont: 'Symbiont') -> None:
        ''' initialize for a simulation event
        Parameters:
            time: time the event is to occur
            event_type: the type of event (see EventType class)
            symbiont: the symbiont driving the event
        '''
        self._time     : float      = time
        self._type     : EventType  = event_type
        self._symbiont : 'Symbiont' = symbiont

    #####################################
    # for using bisect in event list so events are oderable: 
    # http://python3porting.com/problems.html
    def __lt__(self, other: 'Event') -> bool:
        ''' method to compare this event to another, allowing bisect to work
            properly on list of events
        Parameters:
            other: an Event object to compare to
        Returns:
            True if this event should appear before the other event, False o/w
        '''
        # sort first on event time, then on event type
        return (self._time,self._type) < (other._time,other._type)

    #####################################
    ''' simple getter/accessor methods '''
    def getType(self)     -> EventType:  return self._type
    def getTime(self)     -> float:      return self._time 
    def getSymbiont(self) -> 'Symbiont': return self._symbiont

    ###########################
    def __str__(self) -> str:
        ''' returns an str representation of this Event object '''
        return f"{self._type.name} @ t= {self._time} :\n\t{self._symbiont}"

###############################################################################
class EventList:
    ''' class to implement an event list for the simulation model, using a
        Python list to store events in time-sequenced order, using bisect.insort
        to reasonably and efficiently insert events
    '''

    __slots__ = ('_list')

    ###########################
    def __init__(self) -> None:
        ''' initializer, creating an empty event list '''
        self._list = []

    ################################
    def getNextEvent(self) -> 'Event':
        ''' returns the next event to occur in simulated time
        Returns:
            an Event object corresponding to the next event to occur
        '''
        event = None
        if len(self._list) > 0: event = self._list.pop(0)
        return event   # empty list returns None

    ############################################
    def insertEvent(self, event: 'Event') -> None:
        ''' inserts a new event in order of event time into the event list
        Parameters:
            event: an Event object (w/ info time, event type, associated symbiont)
        '''
        assert(event != None)
        # uses __lt__() defined in event.py;
        # note that ties of same time & type will have newest in rightmost position
        bisect.insort(self._list, event) 

    #########################
    def __len__(self) -> int:
        ''' the current length of the event list
        Returns:
            the integer valued number of events in the event list
        '''
        return len(self._list)
