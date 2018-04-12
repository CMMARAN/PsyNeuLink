# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *************************************************  AdaptivePredictionEVCControlMechanism ******************************************************

"""

Overview
--------

An AdaptivePredictionEVCControlMechanism is a `ControlMechanism <ControlMechanism>` that regulates it `ControlSignals <ControlSignal>` in order
to optimize the performance of the System to which it belongs.  AdaptivePredictionEVCControlMechanism is one of the most powerful, but also one
of the most complex components in PsyNeuLink.  It is designed to implement a form of the Expected Value of Control (EVC)
Theory described in `Shenhav et al. (2013) <https://www.ncbi.nlm.nih.gov/pubmed/23889930>`_, which provides useful
background concerning the purpose and structure of the AdaptivePredictionEVCControlMechanism.

An AdaptivePredictionEVCControlMechanism is similar to a standard `ControlMechanism`, with the following exceptions:

  * it can only be assigned to a System as its `controller <System.controller>`, and not in any other capacity
    (see `ControlMechanism_System_Controller`);
  ..
  * it has several specialized functions that are used to search over the `allocations <ControlSignal.allocations>`\\s
    of its its `ControlSignals <ControlSignal>`, and evaluate the performance of its `system <AdaptivePredictionEVCControlMechanism.system>`;
    by default, it simulates its `system <AdaptivePredictionEVCControlMechanism.system>` and evaluates its performance under all combinations
    of ControlSignal values to find the one that optimizes the `Expected Value of Control <AdaptivePredictionEVCControlMechanism_EVC>`, however
    its functions can be customized or replaced to implement other optimization procedures.
  ..
  * it creates a specialized set of `prediction Mechanisms` AdaptivePredictionEVCControlMechanism_Prediction_Mechanisms` that are used to
    simulate the performnace of its `system <AdaptivePredictionEVCControlMechanism.system>`.

.. _AdaptivePredictionEVCControlMechanism_EVC:

Expected Value of Control (EVC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The AdaptivePredictionEVCControlMechanism uses it `function <AdaptivePredictionEVCControlMechanism.function>` to select an `allocation_policy` for its `system
<AdaptivePredictionEVCControlMechanism.system>`.  In the `default configuration <EVC_Default_Configuration>`, an AdaptivePredictionEVCControlMechanism carries out an
exhaustive evaluation of allocation policies, simulating its `system <AdaptivePredictionEVCControlMechanism.system>` under each, and using an
`ObjectiveMechanism` and several `auxiliary functions <AdaptivePredictionEVCControlMechanism_Functions>` to calculate the **expected
value of control (EVC)** for each `allocation_policy`: a cost-benefit analysis that weighs the `cost
<ControlSignal.cost> of the ControlSignals against the outcome of the `system <AdaptivePredictionEVCControlMechanism.system>` \\s performance for
a given `allocation_policy`. The AdaptivePredictionEVCControlMechanism selects the `allocation_policy` that generates the maximum EVC, and
implements that for the next `TRIAL`. Each step of this procedure can be modified, or replaced entirely, by assigning
custom functions to corresponding parameters of the AdaptivePredictionEVCControlMechanism, as described `below <AdaptivePredictionEVCControlMechanism_Functions>`.

.. _AdaptivePredictionEVCControlMechanism_Creation:

Creating an AdaptivePredictionEVCControlMechanism
------------------------

An AdaptivePredictionEVCControlMechanism can be created in any of the ways used to `create a ControlMechanism <ControlMechanism_Creation>`;
it is also created automatically when a `System` is created and the AdaptivePredictionEVCControlMechanism class is specified in the
**controller** argument of the System's constructor (see `System_Creation`).  The ObjectiveMechanism,
the OutputStates it monitors and evaluates, and the parameters controlled by an AdaptivePredictionEVCControlMechanism can be specified in the
standard way for a ControlMechanism (see `ControlMechanism_ObjectiveMechanism` and
`ControlMechanism_Control_Signals`, respectively).

.. note::
   Although an AdaptivePredictionEVCControlMechanism can be created on its own, it can only be assigned to, and executed within a `System` as
   the System's `controller <System.controller>`.

When an AdaptivePredictionEVCControlMechanism is assigned to, or created by a System, it is assigned the OutputStates to be monitored and
parameters to be controlled specified for that System (see `System_Control`), and a `prediction Mechanism
<AdaptivePredictionEVCControlMechanism_Prediction_Mechanisms>` is created for each `ORIGIN` Mechanism in the `system <AdaptivePredictionEVCControlMechanism.system>`.
The prediction Mechanisms are assigned to the AdaptivePredictionEVCControlMechanism's `prediction_mechanisms` attribute. The OutputStates used
to determine an AdaptivePredictionEVCControlMechanism’s allocation_policy and the parameters it controls can be listed using its show method.
The AdaptivePredictionEVCControlMechanism and the Components associated with it in its `system <AdaptivePredictionEVCControlMechanism.system>` can be displayed using
the System's `System.show_graph` method with its **show_control** argument assigned as `True`

An AdaptivePredictionEVCControlMechanism that has been constructed automatically can be customized by assigning values to its attributes (e.g.,
those described above, or its `function <AdaptivePredictionEVCControlMechanism.function>` as described under `EVC_Default_Configuration `below).


.. _AdaptivePredictionEVCControlMechanism_Structure:

Structure
---------

An AdaptivePredictionEVCControlMechanism must belong to a `System` (identified in its `system <AdaptivePredictionEVCControlMechanism.system>` attribute).  In addition
to the standard Components of a `ControlMechanism`, has a specialized set of `prediction mechanisms
<AdaptivePredictionEVCControlMechanism_Prediction_Mechanisms>` and `functions <AdaptivePredictionEVCControlMechanism_Functions>` that it uses to simulate and evaluate
the performance of its `system <AdaptivePredictionEVCControlMechanism.system>` under the influence of different values of its `ControlSignals
<AdaptivePredictionEVCControlMechanism_ControlSignals>`.  Each of these specialized Components is described below.

.. _AdaptivePredictionEVCControlMechanism_Input:

Input
~~~~~

.. _AdaptivePredictionEVCControlMechanism_ObjectiveMechanism:

ObjectiveMechanism
^^^^^^^^^^^^^^^^^^

Like any ControlMechanism, an AdaptivePredictionEVCControlMechanism receives its input from the *OUTCOME* `OutputState
<ObjectiveMechanism_Output>` of an `ObjectiveMechanism`, via a MappingProjection to its `primary InputState
<InputState_Primary>`.  The ObjectiveFunction is listed in the AdaptivePredictionEVCControlMechanism's `objective_mechanism
<AdaptivePredictionEVCControlMechanism.objective_mechanism>` attribute.  By default, the ObjectiveMechanism's function is a `LinearCombination`
function with its `operation <LinearCombination.operation>` attribute assigned as *PRODUCT*;  this takes the product of
the `value <OutputState.value>`\\s of the OutputStates that it monitors (listed in its `monitored_output_states
<ObjectiveMechanism.monitored_output_states>` attribute.  However, this can be customized in a variety of ways:

    * by specifying a different `function <ObjectiveMechanism.function>` for the ObjectiveMechanism
      (see `Objective Mechanism Examples <ObjectiveMechanism_Weights_and_Exponents_Example>` for an example);
    ..
    * using a list to specify the OutputStates to be monitored  (and the `tuples format
      <InputState_Tuple_Specification>` to specify weights and/or exponents for them) in the
      **objective_mechanism** argument of the AdaptivePredictionEVCControlMechanism's constructor;
    ..
    * using the  **monitored_output_states** argument for an ObjectiveMechanism specified in the `objective_mechanism
      <AdaptivePredictionEVCControlMechanism.objective_mechanism>` argument of the EVCMechanism's constructor;
    ..
    * specifying a different `ObjectiveMechanism` in the **objective_mechanism** argument of the AdaptivePredictionEVCControlMechanism's
      constructor. The result of the `objective_mechanism <AdaptivePredictionEVCControlMechanism.objective_mechanism>`'s `function
      <ObjectiveMechanism.function>` is used as the outcome in the calculations described below.

    .. _AdaptivePredictionEVCControlMechanism_Objective_Mechanism_Function_Note:

    .. note::
       If a constructor for an `ObjectiveMechanism` is used for the **objective_mechanism** argument of the
       AdaptivePredictionEVCControlMechanism's constructor, then the default values of its attributes override any used by the AdaptivePredictionEVCControlMechanism
       for its `objective_mechanism <AdaptivePredictionEVCControlMechanism.objective_mechanism>`.  In particular, whereas an AdaptivePredictionEVCControlMechanism uses
       the same default `function <ObjectiveMechanism.function>` as an `ObjectiveMechanism` (`LinearCombination`),
       it uses *PRODUCT* rather than *SUM* as the default value of the `operation <LinearCombination.operation>`
       attribute of the function.  As a consequence, if the constructor for an ObjectiveMechanism is used to specify
       the AdaptivePredictionEVCControlMechanism's **objective_mechanism** argument, and the **operation** argument is not specified,
       *SUM* rather than *PRODUCT* will be used for the ObjectiveMechanism's `function
       <ObjectiveMechanism.function>`.  To ensure that *PRODUCT* is used, it must be specified explicitly in the
       **operation** argument of the constructor for the ObjectiveMechanism (see 1st example under
       `System_Control_Examples`).

The result of the AdaptivePredictionEVCControlMechanism's `objective_mechanism <AdaptivePredictionEVCControlMechanism.objective_mechanism>` is used by its `function
<ObjectiveMechanism.function>` to evaluate the performance of its `system <AdaptivePredictionEVCControlMechanism.system>` when computing the `EVC
<AdaptivePredictionEVCControlMechanism_EVC>`.


.. _AdaptivePredictionEVCControlMechanism_Prediction_Mechanisms:

Prediction Mechanisms
^^^^^^^^^^^^^^^^^^^^^

These are used to provide input to the `system <AdaptivePredictionEVCControlMechanism.system>` when the AdaptivePredictionEVCControlMechanism's default `function
<AdaptivePredictionEVCControlMechanism.function>` (`ControlSignalGridSearch`) `simulates its execution <EVC_Default_Configuration>` to evaluate
the EVC for each `allocation_policy`.  When an AdaptivePredictionEVCControlMechanism is created, a prediction Mechanism is created for each
`ORIGIN` Mechanism in its `system <AdaptivePredictionEVCControlMechanism.system>`, and for each `Projection <Projection>` received by an `ORIGIN`
Mechanism, a `MappingProjection` from the same source is created that projects to the corresponding prediction
Mechanism. The type of `Mechanism <Mechanism>` used for the prediction Mechanisms is specified by the AdaptivePredictionEVCControlMechanism's
`prediction_mechanism_type` attribute, and their parameters can be specified with the `prediction_mechanism_params`
attribute. The default type is an 'TransferMechanism`, that calculates an exponentially weighted time-average of
its input. The prediction mechanisms for an AdaptivePredictionEVCControlMechanism are listed in its `prediction_mechanisms` attribute.


.. _AdaptivePredictionEVCControlMechanism_Functions:

Function
~~~~~~~~

By default, the primary `function <AdaptivePredictionEVCControlMechanism.function>` is `ControlSignalGridSearch` (see
`EVC_Default_Configuration`), that systematically evaluates the effects of its ControlSignals on the performance of
its `system <AdaptivePredictionEVCControlMechanism.system>` to identify an `allocation_policy <AdaptivePredictionEVCControlMechanism.allocation_policy>` that yields the
highest `EVC <AdaptivePredictionEVCControlMechanism_EVC>`.  However, any function can be used that returns an appropriate value (i.e., that
specifies an `allocation_policy` for the number of `ControlSignals <AdaptivePredictionEVCControlMechanism_ControlSignals>` in the AdaptivePredictionEVCControlMechanism's
`control_signals` attribute, using the correct format for the `allocation <ControlSignal.allocation>` value of each
ControlSignal).  In addition to its primary `function <AdaptivePredictionEVCControlMechanism.function>`, an AdaptivePredictionEVCControlMechanism has several auxiliary
functions, that can be used by its `function <AdaptivePredictionEVCControlMechanism.function>` to calculate the EVC to select an
`allocation_policy` with the maximum EVC among a range of policies specified by its ControlSignals.  The default
set of functions and their operation are described in the section that follows;  however, the AdaptivePredictionEVCControlMechanism's
`function <AdaptivePredictionEVCControlMechanism.function>` can call any other function to customize how the EVC is calcualted.

.. _AdaptivePredictionEVCControlMechanism_Default_Configuration:

Default Configuration of EVC Function and its Auxiliary Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In its default configuration, an AdaptivePredictionEVCControlMechanism simulates and evaluates the performance of its `system
<AdaptivePredictionEVCControlMechanism.system>` under a set of allocation_policies determined by the `allocation_samples
<ControlSignal.allocation_samples>` attributes of its `ControlSignals <AdaptivePredictionEVCControlMechanism_ControlSignals>`, and implements
(for the next `TRIAL` of execution) the one that generates the maximum `EVC <AdaptivePredictionEVCControlMechanism_EVC>`.  This is carried out
by the AdaptivePredictionEVCControlMechanism's default `function <AdaptivePredictionEVCControlMechanism.function>` and three auxiliary functions, as described below.

The default `function <AdaptivePredictionEVCControlMechanism.function>` of an AdaptivePredictionEVCControlMechanism is `ControlSignalGridSearch`. It identifies the
`allocation_policy` with the maximum `EVC <AdaptivePredictionEVCControlMechanism_EVC>` by a conducting an exhaustive search over every possible
`allocation_policy`— that is, all combinations of `allocation <ControlSignal.allocation>` values for its `ControlSignals
<AdaptivePredictionEVCControlMechanism_ControlSignals>`, where the `allocation <ControlSignal.allocation>` values sampled for each ControlSignal
are determined by its `allocation_samples` attribute.  For each `allocation_policy`, the AdaptivePredictionEVCControlMechanism executes the
`system <AdaptivePredictionEVCControlMechanism.system>`, evaluates the `EVC <AdaptivePredictionEVCControlMechanism_EVC>` for that policy, and returns the
`allocation_policy` that yields the greatest EVC value. The following steps are used to calculate the EVC in each
`allocation_policy`:

  * **Implement the policy and simulate the System** - assign the `allocation <ControlSignal.allocation>` that the
    selected `allocation_policy` specifies for each ControlSignal, and then simulate the `system <AdaptivePredictionEVCControlMechanism.system>`
    using the corresponding parameter values.
  |
  * **Evaluate the System's performance** - this is carried out by the AdaptivePredictionEVCControlMechanism's `objective_mechanism
    <AdaptivePredictionEVCControlMechanism.objective_mechanism>`, which is executed as part of the simulation of the System.  The `function
    <ObjectiveMechanism.function>` for a default ObjectiveMechanism is a `LinearCombination` Function that combines the
    `value <OutputState.value>`\\s of the OutputStates listed in the AdaptivePredictionEVCControlMechanism's `monitored_output_states
    <AdaptivePredictionEVCControlMechanism.monitored_output_states>` attribute (and the `objective_mechanism
    <AdaptivePredictionEVCControlMechanism.objective_mechanism>`'s `monitored_output_states <ObjectiveMechanism.monitored_output_states>` attribute)
    by taking their elementwise (Hadamard) product.  However, this behavior can be customized in a variety of ways,
    as described `above <AdaptivePredictionEVCControlMechanism_ObjectiveMechanism>`.
  |
  * **Calculate EVC** - call the AdaptivePredictionEVCControlMechanism's `value_function <AdaptivePredictionEVCControlMechanism.value_function>` passing it the
    outcome (received from the `objective_mechanism`) and a list of the `costs <ControlSignal.cost>` \\s of its
    `ControlSignals <AdaptivePredictionEVCControlMechanism_ControlSignals>`.  The default `value_function
    <AdaptivePredictionEVCControlMechanism.value_function>` calls two additional auxiliary functions, in the following order:
    |
    - `cost_function <AdaptivePredictionEVCControlMechanism.cost_function>`, which sums the costs;  this can be configured to weight and/or
      exponentiate individual costs (see `cost_function <AdaptivePredictionEVCControlMechanism.cost_function>` attribute);
    |
    - `combine_outcome_and_cost_function <AdaptivePredictionEVCControlMechanism.combine_outcome_and_cost_function>`, which subtracts the sum of
      the costs from the outcome to generate the EVC;  this too can be configured (see
      `combine_outcome_and_cost_function <AdaptivePredictionEVCControlMechanism.combine_outcome_and_cost_function>`).

In addition to modifying the default functions (as noted above), any or all of them can be replaced with a custom
function to modify how the `allocation_policy <AdaptivePredictionEVCControlMechanism.allocation_policy>` is determined, so long as the custom
function accepts arguments and returns values that are compatible with any other functions that call that function (see
note below).

.. _AdaptivePredictionEVCControlMechanism_Calling_and_Assigning_Functions:

    .. note::
       The `AdaptivePredictionEVCControlMechanism auxiliary functions <AdaptivePredictionEVCControlMechanism_Functions>` described above are all implemented
       as PsyNeuLink `Functions <Function>`.  Therefore, to call a function itself, it must be referenced as
       ``<AdaptivePredictionEVCControlMechanism>.<function_attribute>.function``.  A custom function assigned to one of the auxiliary functions
       can be either a PsyNeuLink `Function <Function>`, or a generic python function or method (including a lambda
       function).  If it is one of the latter, it is automatically "wrapped" as a PsyNeuLink `Function <Function>`
       (specifically, it is assigned as the `function <UserDefinedFunction.function>` attribute of a
       `UserDefinedFunction` object), so that it can be referenced and called in the same manner as
       the default function assignment. Therefore, once assigned, it too must be referenced as
       ``<AdaptivePredictionEVCControlMechanism>.<function_attribute>.function``.

.. _AdaptivePredictionEVCControlMechanism_ControlSignals:

ControlSignals
~~~~~~~~~~~~~~

The OutputStates of an AdaptivePredictionEVCControlMechanism (like any `ControlMechanism`) are a set of `ControlSignals
<ControlSignal>`, that are listed in its `control_signals <AdaptivePredictionEVCControlMechanism.control_signals>` attribute (as well as its
`output_states <ControlMechanism.output_states>` attribute).  Each ControlSignal is assigned a  `ControlProjection`
that projects to the `ParameterState` for a parameter controlled by the AdaptivePredictionEVCControlMechanism.  Each ControlSignal is
assigned an item of the AdaptivePredictionEVCControlMechanism's `allocation_policy`, that determines its `allocation <ControlSignal.allocation>`
for a given `TRIAL` of execution.  The `allocation <ControlSignal.allocation>` is used by a ControlSignal to determine
its `intensity <ControlSignal.intensity>`, which is then assigned as the `value <ControlProjection.value>` of the
ControlSignal's ControlProjection.   The `value <ControlProjection>` of the ControlProjection is used by the
`ParameterState` to which it projects to modify the value of the parameter (see `ControlSignal_Modulation` for
description of how a ControlSignal modulates the value of a parameter it controls).  A ControlSignal also calculates a
`cost <ControlSignal.cost>`, based on its `intensity <ControlSignal.intensity>` and/or its time course. The
`cost <ControlSignal.cost>` is included in the evaluation that the AdaptivePredictionEVCControlMechanism carries out for a given
`allocation_policy`, and that it uses to adapt the ControlSignal's `allocation  <ControlSignal.allocation>` in the
future.  When the AdaptivePredictionEVCControlMechanism chooses an `allocation_policy` to evaluate,  it selects an allocation value from the
ControlSignal's `allocation_samples <ControlSignal.allocation_samples>` attribute.


.. _AdaptivePredictionEVCControlMechanism_Execution:

Execution
---------

An AdaptivePredictionEVCControlMechanism must be the `controller <System.controller>` of a System, and as a consequence it is always the
last `Mechanism <Mechanism>` to be executed in a `TRIAL` for its `system <AdaptivePredictionEVCControlMechanism.system>` (see `System Control
<System_Execution_Control>` and `Execution <System_Execution>`). When an AdaptivePredictionEVCControlMechanism is executed, it updates the
value of its `prediction_mechanisms` and `objective_mechanism`, and then calls its `function <AdaptivePredictionEVCControlMechanism.function>`,
which determines and implements the `allocation_policy` for the next `TRIAL` of its `system <AdaptivePredictionEVCControlMechanism.system>`
\\s execution.  The default `function <AdaptivePredictionEVCControlMechanism.function>` executes the following steps (described in greater
detailed `above <EVC_Default_Configuration>`):

* samples every allocation_policy (i.e., every combination of the `allocation` \\s specified for the AdaptivePredictionEVCControlMechanism's
  ControlSignals specified by their `allocation_samples` attributes);  for each `allocation_policy`, it:

  * Executes the AdaptivePredictionEVCControlMechanism's `system <AdaptivePredictionEVCControlMechanism.system>` with the parameter values specified by that
    `allocation_policy`;  this includes the AdaptivePredictionEVCControlMechanism's `objective_mechanism`, which provides the result
    to the AdaptivePredictionEVCControlMechanism.

  * Calls the AdaptivePredictionEVCControlMechanism's `value_function <AdaptivePredictionEVCControlMechanism.value_function>`, which in turn calls AdaptivePredictionEVCControlMechanism's
    `cost_function <AdaptivePredictionEVCControlMechanism.cost_function>` and `combine_outcome_and_cost_function
    <AdaptivePredictionEVCControlMechanism.combine_outcome_and_cost_function>` to evaluate the EVC for that `allocation_policy`.

  * Selects and returns the `allocation_policy` that generates the maximum EVC value.

This procedure can be modified by specifying a custom function for any or all of the `functions
<AdaptivePredictionEVCControlMechanism_Functions>` referred to above.


.. _AdaptivePredictionEVCControlMechanism_Examples:

Example
-------

The following example implements a System with an AdaptivePredictionEVCControlMechanism (and two processes not shown)::


    >>> import psyneulink as pnl                                                        #doctest: +SKIP
    >>> myRewardProcess = pnl.Process(...)                                              #doctest: +SKIP
    >>> myDecisionProcess = pnl.Process(...)                                            #doctest: +SKIP
    >>> mySystem = pnl.System(processes=[myRewardProcess, myDecisionProcess],           #doctest: +SKIP
    ...                       controller=pnl.AdaptivePredictionEVCControlMechanism,                       #doctest: +SKIP
    ...                       monitor_for_control=[Reward,                              #doctest: +SKIP
    ...                                            pnl.DDM_OUTPUT.DECISION_VARIABLE,    #doctest: +SKIP
    ...                                            (pnl.RESPONSE_TIME, 1, -1)],         #doctest: +SKIP

It uses the System's **monitor_for_control** argument to assign three OutputStates to be monitored.  The first one
references the Reward Mechanism (not shown);  its `primary OutputState <OutputState_Primary>` will be used by default.
The second and third use keywords that are the names of outputStates of a  `DDM` Mechanism (also not shown).
The last one (RESPONSE_TIME) is assigned a weight of 1 and an exponent of -1. As a result, each calculation of the EVC
computation will multiply the value of the primary OutputState of the Reward Mechanism by the value of the
*DDM_DECISION_VARIABLE* OutputState of the DDM Mechanism, and then divide that by the value of the *RESPONSE_TIME*
OutputState of the DDM Mechanism.

See `ObjectiveMechanism <ObjectiveMechanism_Monitored_Output_States_Examples>` for additional examples of how to specify it's
**monitored_output_states** argument, `ControlMechanism <ControlMechanism_Examples>` for additional examples of how to
specify ControlMechanisms, and `System <System_Examples>` for how to specify the `controller <System.controller>`
of a System.

.. _AdaptivePredictionEVCControlMechanism_Class_Reference:

Class Reference
---------------

"""

import numpy as np
import typecheck as tc

from psyneulink.components.component import function_type
from psyneulink.components.functions.function import ModulationParam, _is_modulation_param
from psyneulink.components.mechanisms.adaptive.control.controlmechanism import ControlMechanism
from psyneulink.components.mechanisms.mechanism import MechanismList
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.library.projections.pathway.predictionprojection import PredictionProjection
from psyneulink.components.shellclasses import Function, System_Base
from psyneulink.globals.keywords import COMMAND_LINE, CONTROL, CONTROLLER, COST_FUNCTION, EVC_MECHANISM, FUNCTION, \
    INITIALIZING, INIT_FUNCTION_METHOD_ONLY, PARAMETER_STATES, PREDICTION_MECHANISM, PREDICTION_MECHANISMS, \
    PREDICTION_MECHANISM_PARAMS, PREDICTION_MECHANISM_TYPE, SUM, LEARNING
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.globals.utilities import ContentAddressableList
from psyneulink.library.subsystems.evc.evcauxiliary import ControlSignalGridSearch, ValueFunction
from psyneulink.library.subsystems.evc.evccontrolmechanism import EVCControlMechanism
from psyneulink.scheduling.time import TimeScale

__all__ = [
    'AdaptivePredictionEVCControlMechanism', 'EVCError',
]


class EVCError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class AdaptivePredictionEVCControlMechanism(EVCControlMechanism):
    """AdaptivePredictionEVCControlMechanism(                                                   \
    system=True,                                                       \
    objective_mechanism=None,                                          \
    prediction_mechanism_type=TransferMechanism,                     \
    prediction_mechanism_params=None,                                  \
    function=ControlSignalGridSearch                                   \
    value_function=ValueFunction,                                      \
    cost_function=LinearCombination(operation=SUM),                    \
    combine_outcome_and_cost_function=LinearCombination(operation=SUM) \
    save_all_values_and_policies:bool=:keyword:`False`,                \
    control_signals=None,                                              \
    params=None,                                                       \
    name=None,                                                         \
    prefs=None)

    Subclass of `ControlMechanism <ControlMechanism>` that optimizes the `ControlSignals <ControlSignal>` for a
    `System`.

    COMMENT:
        Class attributes:
            + componentType (str): System Default Mechanism
            + paramClassDefaults (dict):
                + SYSTEM (System)
                + MONITORED_OUTPUT_STATES (list of Mechanisms and/or OutputStates)

        Class methods:
            None

       **********************************************************************************************

       PUT SOME OF THIS STUFF IN ATTRIBUTES, BUT USE DEFAULTS HERE

        # - specification of System:  required param: SYSTEM
        # - kwDefaultController:  True =>
        #         takes over all unassigned ControlProjections (i.e., without a sender) in its System;
        #         does not take monitored states (those are created de-novo)
        # TBI: - CONTROL_PROJECTIONS:
        #         list of projections to add (and for which outputStates should be added)

        # - input_states: one for each performance/environment variable monitored

        ControlProjection Specification:
        #    - wherever a ControlProjection is specified, using kwEVC instead of CONTROL_PROJECTION
        #     this should override the default sender SYSTEM_DEFAULT_CONTROLLER in ControlProjection._instantiate_sender
        #    ? expclitly, in call to "EVC.monitor(input_state, parameter_state=NotImplemented) method

        # - specification of function: default is default allocation policy (BADGER/GUMBY)
        #   constraint:  if specified, number of items in variable must match number of input_states in INPUT_STATES
        #                  and names in list in kwMonitor must match those in INPUT_STATES

       **********************************************************************************************

       NOT CURRENTLY IN USE:

        system : System
            System for which the AdaptivePredictionEVCControlMechanism is the controller;  this is a required parameter.

        default_variable : Optional[number, list or np.ndarray] : `defaultControlAllocation <LINK]>`

    COMMENT


    Arguments
    ---------

    system : System : default None
        specifies the `System` for which the AdaptivePredictionEVCControlMechanism should serve as a `controller <System.controller>`;
        the AdaptivePredictionEVCControlMechanism will inherit any `OutputStates <OutputState>` specified in the **monitor_for_control**
        argument of the `system <AdaptivePredictionEVCControlMechanism.system>`'s constructor, and any `ControlSignals <ControlSignal>`
        specified in its **control_signals** argument.

    objective_mechanism : ObjectiveMechanism, List[OutputState or Tuple[OutputState, list or 1d np.array, list or 1d
    np.array]] : \
    default MonitoredOutputStatesOptions.PRIMARY_OUTPUT_STATES
        specifies either an `ObjectiveMechanism` to use for the AdaptivePredictionEVCControlMechanism or a list of the OutputStates it should
        monitor; if a list of `OutputState specifications <ObjectiveMechanism_Monitored_Output_States>` is used,
        a default ObjectiveMechanism is created and the list is passed to its **monitored_output_states** argument.

    prediction_mechanism_type : CombinationFunction: default TransferMechanism
        the `Mechanism <Mechanism>` class used for `prediction Mechanism(s) <AdaptivePredictionEVCControlMechanism_Prediction_Mechanisms>`.
        Each instance is named using the name of the `ORIGIN` Mechanism + "PREDICTION_MECHANISM"
        and assigned an `OutputState` with a name based on the same.

    prediction_mechanism_params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` passed to the constructor for a Mechanism
        of `prediction_mechanism_type`. The same parameter dictionary is passed to all
        `prediction mechanisms <AdaptivePredictionEVCControlMechanism_Prediction_Mechanisms>` created for the AdaptivePredictionEVCControlMechanism.

    function : function or method : ControlSignalGridSearch
        specifies the function used to determine the `allocation_policy` for the next execution of the
        AdaptivePredictionEVCControlMechanism's `system <AdaptivePredictionEVCControlMechanism.system>` (see `function <AdaptivePredictionEVCControlMechanism.function>` for details).

    value_function : function or method : value_function
        specifies the function used to calculate the `EVC <AdaptivePredictionEVCControlMechanism_EVC>` for the current `allocation_policy`
        (see `value_function <AdaptivePredictionEVCControlMechanism.value_function>` for details).

    cost_function : function or method : LinearCombination(operation=SUM)
        specifies the function used to calculate the cost associated with the current `allocation_policy`
        (see `cost_function <AdaptivePredictionEVCControlMechanism.cost_function>` for details).

    combine_outcome_and_cost_function : function or method : LinearCombination(operation=SUM)
        specifies the function used to combine the outcome and cost associated with the current `allocation_policy`,
        to determine its value (see `combine_outcome_and_cost_function` for details).

    save_all_values_and_policies : bool : default False
        specifes whether to save every `allocation_policy` tested in `EVC_policies` and their values in `EVC_values`.

    control_signals : ControlSignal specification or List[ControlSignal specification, ...]
        specifies the parameters to be controlled by the AdaptivePredictionEVCControlMechanism
        (see `ControlSignal_Specification` for details of specification).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for the
        Mechanism, its `function <AdaptivePredictionEVCControlMechanism.function>`, and/or a custom function and its parameters.  Values
        specified for parameters in the dictionary override any assigned to those parameters in arguments of the
        constructor.

    name : str : default see `name <AdaptivePredictionEVCControlMechanism.name>`
        specifies the name of the AdaptivePredictionEVCControlMechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the AdaptivePredictionEVCControlMechanism; see `prefs <AdaptivePredictionEVCControlMechanism.prefs>` for details.

    Attributes
    ----------

    system : System_Base
        the `System` for which AdaptivePredictionEVCControlMechanism is the `controller <System.controller>`;
        the AdaptivePredictionEVCControlMechanism inherits any `OutputStates <OutputState>` specified in the **monitor_for_control**
        argument of the `system <AdaptivePredictionEVCControlMechanism.system>`'s constructor, and any `ControlSignals <ControlSignal>`
        specified in its **control_signals** argument.

    prediction_mechanisms : List[ProcessingMechanism]
        list of `predictions mechanisms <AdaptivePredictionEVCControlMechanism_Prediction_Mechanisms>` generated for the AdaptivePredictionEVCControlMechanism's
        `system <AdaptivePredictionEVCControlMechanism.system>` when the AdaptivePredictionEVCControlMechanism is created, one for each `ORIGIN` Mechanism in the System.

    origin_prediction_mechanisms : Dict[ProcessingMechanism, ProcessingMechanism]
        dictionary of `prediction mechanisms <AdaptivePredictionEVCControlMechanism_Prediction_Mechanisms>` added to the AdaptivePredictionEVCControlMechanism's
        `system <AdaptivePredictionEVCControlMechanism.system>`, one for each of its `ORIGIN` Mechanisms.  The key for each
        entry is an `ORIGIN` Mechanism of the System, and the value is the corresponding prediction Mechanism.

    prediction_mechanism_type : ProcessingMechanism : default TransferMechanism
        the `ProcessingMechanism <ProcessingMechanism>` class used for `prediction Mechanism(s)
        <AdaptivePredictionEVCControlMechanism_Prediction_Mechanisms>`. Each instance is named based on `ORIGIN` Mechanism +
        "PREDICTION_MECHANISM", and assigned an `OutputState` with a name based on the same.

    prediction_mechanism_params : Dict[param key, param value] : default None
        a `parameter dictionary <ParameterState_Specification>` passed to `prediction_mechanism_type` when
        the `prediction Mechanism <AdaptivePredictionEVCControlMechanism_Prediction_Mechanisms>` is created.  The same dictionary will be passed
        to all instances of `prediction_mechanism_type` created.

    predicted_input : Dict[ProcessingMechanism, value]
        dictionary with the `value <Mechanism_Base.value>` of each `prediction Mechanism
        <AdaptivePredictionEVCControlMechanism_Prediction_Mechanisms>` listed in `prediction_mechanisms` corresponding to each `ORIGIN`
        Mechanism of the System. The key for each entry is the name of an `ORIGIN` Mechanism, and its
        value the `value <Mechanism_Base.value>` of the corresponding prediction Mechanism.

    objective_mechanism : ObjectiveMechanism
        the 'ObjectiveMechanism' used by the AdaptivePredictionEVCControlMechanism to evaluate the performance of its `system
        <AdaptivePredictionEVCControlMechanism.system>`.  If a list of OutputStates is specified in the **objective_mechanism** argument of the
        AdaptivePredictionEVCControlMechanism's constructor, they are assigned as the `monitored_output_states <ObjectiveMechanism.monitored_output_states>`
        attribute for the `objective_mechanism <AdaptivePredictionEVCControlMechanism.objective_mechanism>`.

    monitored_output_states : List[OutputState]
        list of the OutputStates monitored by `objective_mechanism <AdaptivePredictionEVCControlMechanism.objective_mechanism>` (and listed in
        its `monitored_output_states <ObjectiveMechanism.monitored_output_states>` attribute), and used to evaluate the
        performance of the AdaptivePredictionEVCControlMechanism's `system <AdaptivePredictionEVCControlMechanism.system>`.

    COMMENT:
    [TBI]
        monitored_output_states : 3D np.array
            an array of values of the outputStates in `monitored_output_states` (equivalent to the values of
            the AdaptivePredictionEVCControlMechanism's `input_states <AdaptivePredictionEVCControlMechanism.input_states>`).
    COMMENT

    monitored_output_states_weights_and_exponents: List[Tuple[scalar, scalar]]
        a list of tuples, each of which contains the weight and exponent (in that order) for an OutputState in
        `monitored_outputStates`, listed in the same order as the outputStates are listed in `monitored_outputStates`.

    function : function : default ControlSignalGridSearch
        determines the `allocation_policy` to use for the next round of the System's
        execution. The default function, `ControlSignalGridSearch`, conducts an exhaustive (*grid*) search of all
        combinations of the `allocation_samples` of its ControlSignals (and contained in its
        `control_signal_search_space` attribute), by executing the System (using `run_simulation`) for each
        combination, evaluating the result using `value_function`, and returning the `allocation_policy` that yielded
        the greatest `EVC <AdaptivePredictionEVCControlMechanism_EVC>` value (see `AdaptivePredictionEVCControlMechanism_Default_Configuration` for additional details).
        If a custom function is specified, it must accommodate a **controller** argument that specifies an AdaptivePredictionEVCControlMechanism
        (and provides access to its attributes, including `control_signal_search_space`), and must return an array with
        the same format (number and type of elements) as the AdaptivePredictionEVCControlMechanism's `allocation_policy` attribute.

    COMMENT:
        NOTES ON API FOR CUSTOM VERSIONS:
            Gets controller as argument (along with any standard params specified in call)
            Must include **kwargs to receive standard args (variable, params, and context)
            Must return an allocation policy compatible with controller.allocation_policy:
                2d np.array with one array for each allocation value

            Following attributes are available:
            controller._get_simulation_system_inputs gets inputs for a simulated run (using predictionMechanisms)
            controller._assign_simulation_inputs assigns value of prediction_mechanisms to inputs of `ORIGIN` Mechanisms
            controller.run will execute a specified number of trials with the simulation inputs
            controller.monitored_states is a list of the Mechanism OutputStates being monitored for outcome
            controller.input_value is a list of current outcome values (values for monitored_states)
            controller.monitored_output_states_weights_and_exponents is a list of parameterizations for OutputStates
            controller.control_signals is a list of control_signal objects
            controller.control_signal_search_space is a list of all allocationPolicies specifed by allocation_samples
            control_signal.allocation_samples is the set of samples specified for that control_signal
            [TBI:] control_signal.allocation_range is the range that the control_signal value can take
            controller.allocation_policy - holds current allocation_policy
            controller.output_values is a list of current control_signal values
            controller.value_function - calls the three following functions (done explicitly, so each can be specified)
            controller.cost_function - aggregate costs of control signals
            controller.combine_outcome_and_cost_function - combines outcomes and costs
    COMMENT

    value_function : function : default ValueFunction
        calculates the `EVC <AdaptivePredictionEVCControlMechanism_EVC>` for a given `allocation_policy`.  It takes as its arguments an
        `AdaptivePredictionEVCControlMechanism`, an **outcome** value and a list or ndarray of **costs**, uses these to calculate an EVC,
        and returns a three item tuple with the calculated EVC, and the outcome value and aggregated value of costs
        used to calculate the EVC.  The default, `ValueFunction`,  calls the AdaptivePredictionEVCControlMechanism's `cost_function
        <AdaptivePredictionEVCControlMechanism.cost_function>` to aggregate the value of the costs, and then calls its
        `combine_outcome_and_costs <AdaptivePredictionEVCControlMechanism.combine_outcome_and_costs>` to calculate the EVC from the outcome
        and aggregated cost (see `AdaptivePredictionEVCControlMechanism_Default_Configuration` for additional details).  A custom
        function can be assigned to `value_function` so long as it returns a tuple with three items: the calculated
        EVC (which must be a scalar value), and the outcome and cost from which it was calculated (these can be scalar
        values or `None`). If used with the AdaptivePredictionEVCControlMechanism's default `function <AdaptivePredictionEVCControlMechanism.function>`, a custom
        `value_function` must accommodate three arguments (passed by name): a **controller** argument that is the
        AdaptivePredictionEVCControlMechanism for which it is carrying out the calculation; an **outcome** argument that is a value; and a
        `costs` argument that is a list or ndarray.  A custom function assigned to `value_function` can also call any
        of the `helper functions <AdaptivePredictionEVCControlMechanism_Functions>` that it calls (however, see `note
        <AdaptivePredictionEVCControlMechanism_Calling_and_Assigning_Functions>` above).

    cost_function : function : default LinearCombination(operation=SUM)
        calculates the cost of the `ControlSignals <ControlSignal>` for the current `allocation_policy`.  The default
        function sums the `cost <ControlSignal.cost>` of each of the AdaptivePredictionEVCControlMechanism's `ControlSignals
        <AdaptivePredictionEVCControlMechanism_ControlSignals>`.  The `weights <LinearCombination.weights>` and/or `exponents
        <LinearCombination.exponents>` parameters of the function can be used, respectively, to scale and/or
        exponentiate the contribution of each ControlSignal cost to the combined cost.  These must be specified as
        1d arrays in a *WEIGHTS* and/or *EXPONENTS* entry of a `parameter dictionary <ParameterState_Specification>`
        assigned to the **params** argument of the constructor of a `LinearCombination` function; the length of
        each array must equal the number of (and the values listed in the same order as) the ControlSignals in the
        AdaptivePredictionEVCControlMechanism's `control_signals <AdaptivePredictionEVCControlMechanism.control_signals>` attribute. The default function can also be
        replaced with any `custom function <AdaptivePredictionEVCControlMechanism_Calling_and_Assigning_Functions>` that takes an array as
        input and returns a scalar value.  If used with the AdaptivePredictionEVCControlMechanism's default `value_function
        <AdaptivePredictionEVCControlMechanism.value_function>`, a custom `cost_function <AdaptivePredictionEVCControlMechanism.cost_function>` must accommodate two
        arguments (passed by name): a **controller** argument that is the AdaptivePredictionEVCControlMechanism itself;  and a **costs**
        argument that is a 1d array of scalar values specifying the `cost <ControlSignal.cost>` for each ControlSignal
        listed in the `control_signals` attribute of the ControlMechanism specified in the **controller** argument.

    combine_outcome_and_cost_function : function : default LinearCombination(operation=SUM)
        combines the outcome and cost for given `allocation_policy` to determine its `EVC <AdaptivePredictionEVCControlMechanisms_EVC>`. The
        default function subtracts the cost from the outcome, and returns the difference.  This can be modified using
        the `weights <LinearCombination.weights>` and/or `exponents <LinearCombination.exponents>` parameters of the
        function, as described for the `cost_function <AdaptivePredictionEVCControlMechanisms.cost_function>`.  The default function can also be
        replaced with any `custom function <AdaptivePredictionEVCControlMechanism_Calling_and_Assigning_Functions>` that returns a scalar value.  If used with the AdaptivePredictionEVCControlMechanism's default `value_function`, a custom
        If used with the AdaptivePredictionEVCControlMechanism's default `value_function`, a custom combine_outcome_and_cost_function must
        accomoudate three arguments (passed by name): a **controller** argument that is the AdaptivePredictionEVCControlMechanism itself; an
        **outcome** argument that is a 1d array with the outcome of the current `allocation_policy`; and a **cost**
        argument that is 1d array with the cost of the current `allocation_policy`.

    control_signal_search_space : 2d np.array
        an array each item of which is an `allocation_policy`.  By default, it is assigned the set of all possible
        allocation policies, using np.meshgrid to construct all permutations of `ControlSignal` values from the set
        specified for each by its `allocation_samples <AdaptivePredictionEVCControlMechanism.allocation_samples>` attribute.

    EVC_max : 1d np.array with single value
        the maximum `EVC <AdaptivePredictionEVCControlMechanism_EVC>` value over all allocation policies in `control_signal_search_space`.

    EVC_max_state_values : 2d np.array
        an array of the values for the OutputStates in `monitored_output_states` using the `allocation_policy` that
        generated `EVC_max`.

    EVC_max_policy : 1d np.array
        an array of the ControlSignal `intensity <ControlSignal.intensity> values for the allocation policy that
        generated `EVC_max`.

    save_all_values_and_policies : bool : default False
        specifies whether or not to save every `allocation_policy and associated EVC value (in addition to the max).
        If it is specified, each `allocation_policy` tested in the `control_signal_search_space` is saved in
        `EVC_policies`, and their values are saved in `EVC_values`.

    EVC_policies : 2d np.array
        array with every `allocation_policy` tested in `control_signal_search_space`.  The `EVC <AdaptivePredictionEVCControlMechanism_EVC>`
        value of each is stored in `EVC_values`.

    EVC_values :  1d np.array
        array of `EVC <AdaptivePredictionEVCControlMechanism_EVC>` values, each of which corresponds to an `allocation_policy` in `EVC_policies`;

    allocation_policy : 2d np.array : defaultControlAllocation
        determines the value assigned as the `variable <ControlSignal.variable>` for each `ControlSignal` and its
        associated `ControlProjection`.  Each item of the array must be a 1d array (usually containing a scalar)
        that specifies an `allocation` for the corresponding ControlSignal, and the number of items must equal the
        number of ControlSignals in the AdaptivePredictionEVCControlMechanism's `control_signals` attribute.

    control_signals : ContentAddressableList[ControlSignal]
        list of the AdaptivePredictionEVCControlMechanism's `ControlSignals <AdaptivePredictionEVCControlMechanism_ControlSignals>`, including any that it inherited
        from its `system <AdaptivePredictionEVCControlMechanism.system>` (same as the AdaptivePredictionEVCControlMechanism's `output_states
        <Mechanism_Base.output_states>` attribute); each sends a `ControlProjection` to the `ParameterState` for the
        parameter it controls

    name : str
        the name of the AdaptivePredictionEVCControlMechanism; if it is not specified in the **name** argument of the constructor, a
        default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the AdaptivePredictionEVCControlMechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

    """

    componentType = EVC_MECHANISM
    initMethod = INIT_FUNCTION_METHOD_ONLY


    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to Type automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'DefaultControlMechanismCustomClassPreferences',
    #     kp<pref>: <setting>...}

    from psyneulink.components.functions.function import LinearCombination
    # from Components.__init__ import DefaultSystem
    paramClassDefaults = ControlMechanism.paramClassDefaults.copy()
    paramClassDefaults.update({PARAMETER_STATES: NotImplemented}) # This suppresses parameterStates

    @tc.typecheck
    def __init__(self,
                 system:tc.optional(System_Base)=None,
                 objective_mechanism:tc.optional(tc.any(ObjectiveMechanism, list))=None,
                 prediction_mechanism_type=TransferMechanism,
                 prediction_mechanism_params:tc.optional(dict)=None,
                 control_signals:tc.optional(list) = None,
                 modulation:tc.optional(_is_modulation_param)=ModulationParam.MULTIPLICATIVE,
                 function=ControlSignalGridSearch,
                 value_function=ValueFunction,
                 cost_function=LinearCombination(operation=SUM,
                                                 context=componentType+COST_FUNCTION),
                 combine_outcome_and_cost_function=LinearCombination(operation=SUM,
                                                                     context=componentType+FUNCTION),
                 save_all_values_and_policies:bool=False,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=componentType+INITIALIZING):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(system=system,
                                                  prediction_mechanism_type=prediction_mechanism_type,
                                                  prediction_mechanism_params=prediction_mechanism_params,
                                                  objective_mechanism=objective_mechanism,
                                                  function=function,
                                                  control_signals=control_signals,
                                                  modulation=modulation,
                                                  value_function=value_function,
                                                  cost_function=cost_function,
                                                  combine_outcome_and_cost_function=combine_outcome_and_cost_function,
                                                  save_all_values_and_policies=save_all_values_and_policies,
                                                  params=params)

        super(AdaptivePredictionEVCControlMechanism, self).__init__(# default_variable=default_variable,
                                           # size=size,
                                           system=system,
                                           objective_mechanism=objective_mechanism,
                                           function=function,
                                           control_signals=control_signals,
                                           modulation=modulation,
                                           params=params,
                                           name=name,
                                           prefs=prefs,
                                           context=self)

    def _instantiate_prediction_mechanisms(self, system:System_Base, context=None):
        """Add prediction Mechanism and associated process for each `ORIGIN` (input) Mechanism in system

        Instantiate prediction_mechanisms for `ORIGIN` Mechanisms in system;

        For each `ORIGIN` Mechanism in system:
            - instantiate a corresponding predictionMechanism
            - instantiate a Process, with a pathway that projects from the ORIGIN to the prediction Mechanism
            - add the Process to system.processes

        Instantiate self.predicted_input dict:
            - key for each entry is an `ORIGIN` Mechanism of system
            - value of each entry is the value of the corresponding predictionMechanism:
                each value is a 2d array, each item of which is the value of an InputState of the predictionMechanism

        Args:
            context:
        """

        if hasattr(system, CONTROLLER) and hasattr(system.controller, PREDICTION_MECHANISMS):
            # If system's controller already has prediction_mechanisms, and origin mechanisms have not changed
            if set(self.origin_prediction_mechanisms.keys()) is set(system.origin_mechanisms.mechanisms):
                self.prediction_mechanisms = system.controller.prediction_mechanisms
                self.origin_prediction_mechanisms = system.controller.origin_prediction_mechanisms
                self.predicted_input = system.controller.predicted_input
            return

        # Dictionary of prediction_mechanisms, keyed by the ORIGIN Mechanism to which they correspond
        self.origin_prediction_mechanisms = {}

        # List of prediction Mechanism tuples (used by system to execute them)
        self.prediction_mechs = []

        # Get any params specified for predictionMechanism(s) by AdaptivePredictionEVCControlMechanism
        try:
            prediction_mechanism_params = self.paramsCurrent[PREDICTION_MECHANISM_PARAMS]
        except KeyError:
            prediction_mechanism_params = {}

        for origin_mech in system.origin_mechanisms.mechanisms:
            state_names = []
            variable = []
            for state_name in origin_mech.input_states.names:
                state_names.append(state_name)
                variable.append(origin_mech.input_states[state_name].instance_defaults.variable)

            prediction_input_mechanism = TransferMechanism(name=origin_mech.name + " Prediction Input Mechanism")

            # Instantiate PredictionMechanism
            prediction_mechanism = self.paramsCurrent[PREDICTION_MECHANISM_TYPE](
                    name=origin_mech.name + " " + PREDICTION_MECHANISM,
                    default_variable=variable,
                    input_states=state_names,
                    integrator_mode=True,
                    params = prediction_mechanism_params,
                    context=context,
            )

            prediction_mechanism._role = CONTROL
            prediction_mechanism.origin_mech = origin_mech

            # prediction_input_mechanism._role = CONTROL
            # prediction_input_mechanism.origin_mech = origin_mech

            # Assign projections TO prediction_mechanism that duplicate those received by origin_mech
            #    (this includes those from ProcessInputState, SystemInputState and/or recurrent ones
            # Should only be executed during processing!
            for orig_input_state in origin_mech.input_states:
                # must copy path afferents in order to not create an infinite loop when we add prediction projections!
                original_path_afferents = orig_input_state.path_afferents.copy()

                for projection in original_path_afferents:
                    # projections from system/process input states to prediction input mechanisms
                    MappingProjection(sender=projection.sender,
                                      # receiver=prediction_mechanism,
                                      receiver=prediction_input_mechanism,
                                      matrix=projection.matrix)


                # projections from prediction input mechanisms to prediction mechanisms
                MappingProjection(sender=prediction_input_mechanism,
                                  receiver=prediction_mechanism)

                # # projections from prediction mechanisms to origin mechanisms
                # PredictionProjection(sender=prediction_mechanism,
                #                      receiver=orig_input_state)

                # tuple specfication of learning: matrix=(projection.matrix, LEARNING)

                # MappingProjection(name="prediction_projection",
                #                   sender=prediction_mechanism,
                #                   receiver=orig_input_state,
                #                   matrix=(projection.matrix, LEARNING))

            # # FIX: REPLACE REFERENCE TO THIS ELSEWHERE WITH REFERENCE TO MECH_TUPLES BELOW
            self.origin_prediction_mechanisms[origin_mech] = prediction_mechanism

            # Add to list of AdaptivePredictionEVCControlMechanism's prediction_object_items
            # prediction_object_item = prediction_mechanism
            self.prediction_mechs.append(prediction_mechanism)

            # Add to system execution_graph and execution_list

            system.execution_graph[prediction_input_mechanism] = set()
            system.execution_list.append(prediction_input_mechanism)

            system.execution_graph[prediction_mechanism] = set()
            system.execution_list.append(prediction_mechanism)

            print(system.execution_list)
        self.prediction_mechanisms = MechanismList(self, self.prediction_mechs)

        # Assign list of destinations for predicted_inputs:
        #    the variable of the ORIGIN Mechanism for each Process in the system
        self.predicted_input = {}
        for i, origin_mech in zip(range(len(system.origin_mechanisms)), system.origin_mechanisms):
            self.predicted_input[origin_mech] = system.processes[i].origin_mechanisms[0].instance_defaults.variable

    def _execute(self,
                    variable=None,
                    runtime_params=None,
                    context=None):
        if not 'System.controller setter' in context: # cxt-test
            self._update_predicted_input()
        # self.system._cache_state()

        # CONSTRUCT SEARCH SPACE

        control_signal_sample_lists = []
        control_signals = self.control_signals

        # Get allocation_samples for all ControlSignals
        num_control_signals = len(control_signals)

        for control_signal in self.control_signals:
            control_signal_sample_lists.append(control_signal.allocation_samples)

        # Construct control_signal_search_space:  set of all permutations of ControlProjection allocations
        #                                     (one sample from the allocationSample of each ControlProjection)
        # Reference for implementation below:
        # http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
        self.control_signal_search_space = \
            np.array(np.meshgrid(*control_signal_sample_lists)).T.reshape(-1, num_control_signals)

        # EXECUTE SEARCH

        # IMPLEMENTATION NOTE:
        # self.system._store_system_state()

        allocation_policy = self.function(controller=self,
                                          variable=variable,
                                          runtime_params=runtime_params,
                                          context=context)
        # IMPLEMENTATION NOTE:
        # self.system._restore_system_state()

        return allocation_policy

    def _update_predicted_input(self):
        """Assign values of prediction mechanisms to predicted_input

        Assign value of each predictionMechanism.value to corresponding item of self.predictedIinput
        Note: must be assigned in order of self.system.processes

        """

        placeholder_inputs = {}
        for origin_mech in self.system.origin_mechanisms:
            # placeholder_inputs[origin_mech] = 0.0*origin_mech.instance_defaults.variable
            prediction_mechanism = self.origin_prediction_mechanisms[origin_mech]
            placeholder_inputs[origin_mech] = prediction_mechanism.value
        self.predicted_input = placeholder_inputs

    def run_simulation(self,
                       inputs,
                       allocation_vector,
                       runtime_params=None,
                       context=None):
        """
        Run simulation of `System` for which the AdaptivePredictionEVCControlMechanism is the `controller <System.controller>`.

        Arguments
        ----------

        allocation_vector : (1D np.array)
            the allocation policy to use in running the simulation, with one allocation value for each of the
            AdaptivePredictionEVCControlMechanism's ControlSignals (listed in `control_signals`).

        runtime_params : Optional[Dict[str, Dict[str, Dict[str, value]]]]
            a dictionary that can include any of the parameters used as arguments to instantiate the mechanisms,
            their functions, or Projection(s) to any of their states.  See `Mechanism_Runtime_Parameters` for a full
            description.

        """

        # original_smoothing_factors = {}
        # for prediction_mechanism in self.prediction_mechanisms:
        #     original_smoothing_factors[prediction_mechanism] = prediction_mechanism.smoothing_factor
        #     prediction_mechanism.smoothing_factor = 0.0


        if self.value is None:
            # Initialize value if it is None
            self.value = np.empty(len(self.control_signals))

        # Implement the current allocation_policy over ControlSignals (outputStates),
        #    by assigning allocation values to AdaptivePredictionEVCControlMechanism.value, and then calling _update_output_states
        for i in range(len(self.control_signals)):
            # self.control_signals[list(self.control_signals.values())[i]].value = np.atleast_1d(allocation_vector[i])
            self.value[i] = np.atleast_1d(allocation_vector[i])
        self._update_output_states(runtime_params=runtime_params, context=context)

        self.system.run(inputs=inputs, context=context)

        # Get outcomes for current allocation_policy
        #    = the values of the monitored output states (self.input_states)
        # self.objective_mechanism.execute(context=EVC_SIMULATION)
        monitored_states = self._update_input_states(runtime_params=runtime_params, context=context)

        for i in range(len(self.control_signals)):
            self.control_signal_costs[i] = self.control_signals[i].cost

        # for prediction_mechanism in self.prediction_mechanisms:
        #     prediction_mechanism.smoothing_factor = original_smoothing_factors[prediction_mechanism]

        return monitored_states

