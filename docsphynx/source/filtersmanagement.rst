Filter Management Module
========================
The module contains one single class which manages all the trackers connected to each track  by the track manager.

.. automodule:: filter_management

FilterManager
-------------
FilterManager is a class of which primary task is to organize life of all trackers :meth:`tracking_filters.TrackingFilter`
working alongside with their Tracks :meth:`data_containers.Track`. The FilterManager is directly controlled by a TrackManager :meth:`track_management.TrackManager`.
In the same time as a new Track is initialized and its TrackID is assigned a new Tracker is started with the same TrackID


The main part of the FilterManager is a list of all active filters - trackers where every tracker is a representation
of one of the filter - tracker type from a file :doc:`tracking_filters.py </trackingfilters>` file.

.. autoclass:: FilterManager
    :members:
