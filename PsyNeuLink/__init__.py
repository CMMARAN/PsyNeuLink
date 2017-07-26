# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ***********************************************  Init ****************************************************************
import logging


# https://stackoverflow.com/a/17276457/3131666
class Whitelist(logging.Filter):
    def __init__(self, *whitelist):
        self.whitelist = [logging.Filter(name) for name in whitelist]

    def filter(self, record):
        return any(f.filter(record) for f in self.whitelist)


class Blacklist(Whitelist):
    def filter(self, record):
        return not Whitelist.filter(self, record)


logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
for handler in logging.root.handlers:
    handler.addFilter(Blacklist(
        'PsyNeuLink.Scheduling.Scheduler',
        'PsyNeuLink.Scheduling.Condition',
    ))

__all__ = ['System',
           'system',
           'process',
           'CentralClock',
           'TransferMechanism',
           'IntegratorMechanism',
           'DDM',
           'EVCMechanism',
           'ComparatorMechanism',
           'WeightedErrorMechanism',
           'MappingProjection',
           'ControlProjection',
           'LearningProjection',
           'UserDefinedFunction',
           'LinearCombination',
           'Linear',
           'Exponential',
           'Logistic',
           'SoftMax',
           'Integrator',
           'LinearMatrix',
           'BackPropagation',
           'FunctionOutputType',
           'FUNCTION',
           'FUNCTION_PARAMS',
           'INPUT_STATES',
           'PARAMETER_STATES',
           'OUTPUT_STATES',
           'MAKE_DEFAULT_CONTROLLER',
           'MONITOR_FOR_CONTROL',
           'INITIALIZER',
           'WEIGHTS',
           'EXPONENTS',
           'OPERATION',
           'OFFSET',
           'SCALE',
           'MATRIX',
           'IDENTITY_MATRIX',
           'HOLLOW_MATRIX',
           'FULL_CONNECTIVITY_MATRIX',
           'DEFAULT_MATRIX',
           'ALL',
           'MAX_VAL',
           'MAX_INDICATOR',
           'PROB']
