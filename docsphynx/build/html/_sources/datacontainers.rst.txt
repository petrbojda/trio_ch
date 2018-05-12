.. _TrackPoint-label:

Data Containers Module
======================
The module defines all the necessary containers to store, sort, filter, import and export data
either raw radar detections or already processed tracks. Also available here are classes to keep
and manipulate data from a referential DGPS system.

All the containers are based on lists of appropriate points. Lists are inherited from a python's built-in
class 'list' with additional methods. Points are inherited from a basic 'object' class.

.. module:: data_containers

DetectionPoint
--------------
Defines a class which represents a raw radar detection.

.. autoclass:: DetectionPoint
    :members:

    .. automethod:: __init__

ReferencePoint
--------------

.. autoclass:: ReferencePoint
    :members:

    .. automethod:: __init__

TrackPoint
----------

.. autoclass:: TrackPoint
    :members:

    .. automethod:: __init__


DetectionList
-------------

.. autoclass:: DetectionList
    :members:

    .. automethod:: __init__

UnAssignedDetectionList
-----------------------

.. autoclass:: UnAssignedDetectionList
    :members:

    .. automethod:: __init__

Track
-----

.. autoclass:: Track
    :members:

    .. automethod:: __init__
