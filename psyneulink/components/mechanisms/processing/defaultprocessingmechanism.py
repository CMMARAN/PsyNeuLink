# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **********************************************  Mechanism ***********************************************************

"""
**[DOCUMENTATION STILL UNDER CONSTRUCTION]**

"""
import numpy as np
import typecheck as tc

from psyneulink.components.mechanisms.mechanism import Mechanism_Base
from psyneulink.globals.context import ContextFlags
from psyneulink.globals.defaults import SystemDefaultInputValue
from psyneulink.globals.keywords import DEFAULT_PROCESSING_MECHANISM, FUNCTION, FUNCTION_PARAMS, INTERCEPT, SLOPE
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.globals.preferences.preferenceset import PreferenceLevel

# **************************************** DefaultProcessingMechanism ******************************************************

__all__ = []


# class DefaultProcessingMechanism_Base(ProcessingMechanism_Base):
class DefaultProcessingMechanism_Base(Mechanism_Base):
    """Subclass of `ProcessingMechanism <ProcessingMechanism>` used to implement SystemDefaultInputMechanism,
    DefaultControlMechanism, and SystemDefaultOutputMechanism.

    Description:
        Implements "dummy" Mechanism used to implement default input, control signals, and outputs to other mechanisms

    Class attributes:
        + componentType (str): System Default Mechanism
        + paramClassDefaults (dict):
            # + kwInputStateValue: [0]
            # + kwOutputStateValue: [1]
            + FUNCTION: Linear
            + FUNCTION_PARAMS:{SLOPE:1, INTERCEPT:0}
    """

    componentName = DEFAULT_PROCESSING_MECHANISM
    componentType = DEFAULT_PROCESSING_MECHANISM
    onlyFunctionOnInit = True

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # Any preferences specified below will override those specified in SubtypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to SUBTYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'DefaultProcessingMechanismClassPreferences',
    #     kp<pref>: <setting>...}

    class ClassDefaults(Mechanism_Base.ClassDefaults):
        variable = np.array([SystemDefaultInputValue])

    from psyneulink.components.functions.function import Linear
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        FUNCTION: Linear,
        FUNCTION_PARAMS: {SLOPE: 1, INTERCEPT: 0}
    })

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 function=None,
                 ):
        """Add Linear as default function, assign default name, and call super.__init__

        :param default_variable: (value)
        :param size: (int or list/array of ints)
        :param params: (dict)
        :param name: (str)
        :param prefs: (PreferenceSet)
        """

        params = self._assign_args_to_param_dicts(params=params)

        super(DefaultProcessingMechanism_Base, self).__init__(default_variable=default_variable,
                                                              size=size,
                                                              function=function,
                                                              params=params,
                                                              name=name,
                                                              prefs=prefs,
                                                              context=ContextFlags.CONSTRUCTOR)
