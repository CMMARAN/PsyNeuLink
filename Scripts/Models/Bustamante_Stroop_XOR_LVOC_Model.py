# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* Bustamante_Stroop_XOR_LVOC_Model ***************************************

'''
Implements a model of the `Stroop XOR task
<https://scholar.google.com/scholar?hl=en&as_sdt=0%2C31&q=laura+bustamante+cohen+musslick&btnG=>`_
using a version of the `Learned Value of Control Model
<https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006043&rev=2>`_
'''

import psyneulink as pnl
import numpy as np

np.random.seed(0)

def w_fct(stim, color_control):
    '''function for word_task, to modulate strength of word reading based on 1-strength of color_naming ControlSignal'''
    return stim * (1-color_control)
w_fct_UDF = pnl.UserDefinedFunction(custom_function=w_fct, color_control=1)


def objective_function(v):
    '''function used for ObjectiveMechanism of lvoc
     v[0] = output of DDM: [probability of color naming, probability of word reading]
     v[1] = reward:        [color naming rewarded, word reading rewarded]
     '''
    return np.sum(v[0]*v[1])


color_stim = pnl.TransferMechanism(name='Color Stimulus', size=8)
word_stim = pnl.TransferMechanism(name='Word Stimulus', size=8)

color_task = pnl.TransferMechanism(name='Color Task')
word_task = pnl.ProcessingMechanism(name='Word Task', function=w_fct_UDF)

reward = pnl.TransferMechanism(name='Reward', size=2)

task_decision = pnl.DDM(name='Task Decision',
            # function=pnl.NavarroAndFuss,
            output_states=[pnl.DDM_OUTPUT.PROBABILITY_UPPER_THRESHOLD,
                           pnl.DDM_OUTPUT.PROBABILITY_LOWER_THRESHOLD])

lvoc = pnl.LVOCControlMechanism(name='LVOC ControlMechanism',
                                feature_predictors={pnl.SHADOW_EXTERNAL_INPUTS:[color_stim, word_stim]},
                                function=pnl.BayesGLM(),
                                objective_mechanism=pnl.ObjectiveMechanism(name='LVOC ObjectiveMechanism',
                                                                           monitored_output_states=[task_decision,
                                                                                                    reward],
                                                                           function=objective_function),
                                prediction_terms=[pnl.PV.FC, pnl.PV.COST],
                                update_rate=1.0,
                                terminal_objective_mechanism=True,
                                # control_signals={'COLOR CONTROL':[(pnl.SLOPE, color_task),
                                #                                    ('color_control', word_task)]}
                                # control_signals={pnl.NAME:'COLOR CONTROL',
                                #                  pnl.PROJECTIONS:[(pnl.SLOPE, color_task),
                                #                                   ('color_control', word_task)],
                                #                  pnl.COST_OPTIONS:[pnl.ControlSignalCosts.INTENSITY,
                                #                                    pnl.ControlSignalCosts.ADJUSTMENT],
                                #                  pnl.INTENSITY_COST_FUNCTION:pnl.Exponential(rate=0.25, bias=-3),
                                #                  pnl.ADJUSTMENT_COST_FUNCTION:pnl.Exponential(rate=0.25,bias=-3)}
                                control_signals=pnl.ControlSignal(projections=[(pnl.SLOPE, color_task),
                                                                               ('color_control', word_task)],
                                                                  # function=pnl.ReLU,
                                                                  function=pnl.Logistic,
                                                                  cost_options=[pnl.ControlSignalCosts.INTENSITY,
                                                                                pnl.ControlSignalCosts.ADJUSTMENT],
                                                                  intensity_cost_function=pnl.Exponential(rate=0.25,
                                                                                                          bias=-3),
                                                                  adjustment_cost_function=pnl.Exponential(rate=0.25,
                                                                                                           bias=-3)
                                                                  )
                                )
c = pnl.Composition(name='Stroop XOR Model')
c.add_c_node(color_stim)
c.add_c_node(word_stim)
c.add_c_node(color_task, required_roles=pnl.CNodeRole.ORIGIN)
c.add_c_node(word_task, required_roles=pnl.CNodeRole.ORIGIN)
c.add_c_node(reward)
c.add_c_node(task_decision)
c.add_projection(sender=color_task, receiver=task_decision)
c.add_projection(sender=word_task, receiver=task_decision)
c.add_c_node(lvoc)

# c.show_graph()

input_dict = {color_stim:[[1,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0]],
              word_stim: [[1,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0]],
              color_task:[[1], [1]],
              word_task: [[-1], [-1]],
              reward:    [[1,0], [1,0]]}
c.run(inputs=input_dict)

