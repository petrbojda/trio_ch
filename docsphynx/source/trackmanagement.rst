Track Management Module
=======================
The module contains one single class which manages all the necessary processes to organize all tracks, starts them from
the scratch and stops them accordingly. Here is being kept a list of current tracks, they are updated when a new detection is
assigned to them, incoming detections are sorted out and either assigned to already existing track or they are stored into
a list of unassigned detections.

.. automodule:: track_management

TrackManager
------------
TrackManager is a class of which primary task is to organize life of all tracks starting by their initiation to
the moment when they are terminated. Advanced management as a parallel tracks merging is not implemented yet.

The main part of the TrackManager is a list of all active tracks where every track is a list itself, see
:meth:`data_containers.Track` in a file :doc:`data_containers.py </datacontainers>` file.

Another list is a list of unassigned detections, :meth:`data_containers.UnAssignedDetectionList` defined in a
:doc:`data_containers.py </datacontainers>` file. A newly incoming detection if not assigned to any existing track is
stored here. Every newcomer is then tested if it can form a new track in combination with other detections already
in a list. If not they are removed from a list and 'trashed' after couple of cycles in order to free the memory and avoid
any possible false track initiation.

.. autoclass:: TrackManager
    :members:
