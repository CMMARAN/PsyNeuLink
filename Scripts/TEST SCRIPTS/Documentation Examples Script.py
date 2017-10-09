#
# Runs examples in the PsyNeuLink Documentation
#

from psyneulink.components.functions.function import *
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.process import Process
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.library.mechanisms.processing.integrator import ddm

# from PsyNeuLink.Components.Functions.Function import Logistic, random_matrix

#region PROCESS EXAMPLES ********************************************************************

# PARAMETER SPECIFICATIONS
# Tests following specifications for a parameter and associated parameterStates:
#     Mechanism
#     - value
#     - function
#     - ControlProjection name
#     - LearningProjection name
#     - ControlProjection KEYWORD
#     - LearningProjection KEYWORD
#     - ControlProjection class
#     - LearningProjection class
#     - ControlProjection constructor (with params)
#     - LearningProjection constructor (with params)
#     - (value, ControlProjection name)
#     - (value, LearningProjection name)
#     - (value, ControlProjection KEYWORD)
#     - (value, LearningProjection KEYWORD)
#     - (value, ControlProjection class)
#     - (value, LearningProjection class)
#     - (value, ControlProjection constructor (with params))
#     - (value, LearningProjection constructor (with params))
#     - (function, ControlProjection name)
#     - (function, LearningProjection name)
#     - (function, ControlProjection KEYWORD)
#     - (function, LearningProjection KEYWORD)
#     - (function, ControlProjection class)
#     - (function, LearningProjection class)
#     - (function, ControlProjection constructor (with params))
#     - (function, LearningProjection constructor (with params))
#     Function (within a mechanism specification)
#     - value
#     - function
#     - ControlProjection name
#     - LearningProjection name
#     - ControlProjection KEYWORD
#     - LearningProjection KEYWORD
#     - ControlProjection class
#     - LearningProjection class
#     - ControlProjection constructor (with params)
#     - LearningProjection constructor (with params)
#     - (value, ControlProjection name)
#     - (value, LearningProjection name)
#     - (value, ControlProjection KEYWORD)
#     - (value, LearningProjection KEYWORD)
#     - (value, ControlProjection class)
#     - (value, LearningProjection class)
#     - (value, ControlProjection constructor (with params))
#     - (value, LearningProjection constructor (with params))
#     - (function, ControlProjection name)
#     - (function, LearningProjection name)
#     - (function, ControlProjection KEYWORD)
#     - (function, LearningProjection KEYWORD)
#     - (function, ControlProjection class)
#     - (function, LearningProjection class)
#     - (function, ControlProjection constructor (with params))
#     - (function, LearningProjection constructor (with params))
#     MappingProjection
#     - value
#     - function
#     - ControlProjection name
#     - LearningProjection name
#     - ControlProjection KEYWORD
#     - LearningProjection KEYWORD
#     - ControlProjection class
#     - LearningProjection class
#     - ControlProjection constructor (with params)
#     - LearningProjection constructor (with params)
#     - (value, ControlProjection name)
#     - (value, LearningProjection name)
#     - (value, ControlProjection KEYWORD)
#     - (value, LearningProjection KEYWORD)
#     - (value, ControlProjection class)
#     - (value, LearningProjection class)
#     - (value, ControlProjection constructor (with params))
#     - (value, LearningProjection constructor (with params))
#     - (function, ControlProjection name)
#     - (function, LearningProjection name)
#     - (function, ControlProjection KEYWORD)
#     - (function, LearningProjection KEYWORD)
#     - (function, ControlProjection class)
#     - (function, LearningProjection class)
#     - (function, ControlProjection constructor (with params))
#     - (function, LearningProjection constructor (with params))

# my_test_mechanism = TransferMechanism(function=Logistic(bias=99,
#                                                         gain=ControlProjection()),
#                                       noise=0.3,
#                                       name='MY_TRANSFER_MECH'
#                                       )


# Specification of mechanisms in a pathway
mechanism_1 = TransferMechanism()
mechanism_2 = ddm()
some_params = {PARAMETER_STATE_PARAMS:{THRESHOLD:2,NOISE:0.1}}
my_process = Process(pathway=[mechanism_1, TransferMechanism, (mechanism_2, some_params)])
print(my_process.execute())

# Default projection specification
mechanism_1 = TransferMechanism()
mechanism_2 = TransferMechanism()
mechanism_3 = ddm()
my_process = Process(pathway=[mechanism_1, mechanism_2, mechanism_3])
print(my_process.execute())

# Inline projection specification using an existing projection
mechanism_1 = TransferMechanism()
mechanism_2 = TransferMechanism()
mechanism_3 = ddm()
projection_A = MappingProjection()
my_process = Process(pathway=[mechanism_1, projection_A, mechanism_2, mechanism_3])
print(my_process.execute())

mechanism_1 = TransferMechanism()
mechanism_2 = TransferMechanism()
mechanism_3 = ddm()
# Inline projection specification using a keyword
my_process = Process(pathway=[mechanism_1, RANDOM_CONNECTIVITY_MATRIX, mechanism_2, mechanism_3])
print(my_process.execute())

# Stand-alone projection specification
mechanism_1 = TransferMechanism()
mechanism_2 = TransferMechanism()
mechanism_3 = ddm()
projection_A = MappingProjection(sender=mechanism_1, receiver=mechanism_2)
my_process = Process(pathway=[mechanism_1, mechanism_2, mechanism_3])
print(my_process.execute())

# Process that implements learning
mechanism_1 = TransferMechanism(function=Logistic)
mechanism_2 = TransferMechanism(function=Logistic)
mechanism_3 = TransferMechanism(function=Logistic)
# my_process = Process(pathway=[mechanism_1, mechanism_2, mechanism_3],
#                      learning=LEARNING_PROJECTION)
my_process = Process(pathway=[mechanism_1, mechanism_2, mechanism_3],
                     learning=LEARNING_PROJECTION,
                     target=[0])
print(my_process.execute())

#endregion


#region SYSTEM EXAMPLES ********************************************************************
#
#endregion