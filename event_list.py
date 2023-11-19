from enum import Enum
from heapq import heappush, heappop

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
    __slots__ = ('_time', '_type', '_symbiont', '_event_num')

    # class-level variable to track the number of total eventds
    _event_cnt : int = 0

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
        self._event_num: int        = Event._event_cnt

        Event._event_cnt += 1

    #####################################
    def __lt__(self, other: 'Event') -> bool:
        ''' method to compare this event to another
        Parameters:
            other: an Event object to compare to
        Returns:
            True if this event should appear before the other event, False o/w
        '''
        # sort first on event time, then on event type, then on event number (JIC)
        return (self._time,self._type,self._event_num) \
            < (other._time,other._type,other._event_num)

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
    ''' Class to implement an event list for the simulation model, using a
        Python list initially, but converted to a priority queue using
        heapq.heappush and heapq.heappop.  This facilitates efficient insertion
        and removal of time-sequenced events as part of the event calendar 
        (event list).
    '''
    __slots__ = ('_heap')

    ###########################
    def __init__(self) -> None:
        ''' initializer, creating an empty event list (for heap) '''
        self._heap = []

    ##########################################
    def getNextEvent(self) -> 'Event or None':
        ''' returns the next event to occur in simulated time
        Returns:
            an Event object corresponding to the next event to occur
        '''
        event = None
        if len(self._heap) > 0: event = heappop(self._heap)
        return event   # empty list returns None

    ##############################################
    def insertEvent(self, event: 'Event') -> None:
        ''' inserts a new event in order of event time into the event list
        Parameters:
            event: an Event object (w/ info time, event type, associated symbiont)
        '''
        assert(event != None)
        heappush(self._heap, event)

    #########################
    def __len__(self) -> int:
        ''' the current length of the event list
        Returns:
            the integer valued number of events in the event list
        '''
        return len(self._heap)
