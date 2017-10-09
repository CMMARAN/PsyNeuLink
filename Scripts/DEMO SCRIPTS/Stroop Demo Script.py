# from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.Deprecated.LinearMechanism import *

from psyneulink.components.mechanisms.processing.transfermechanism import *
from psyneulink.components.process import Process
from psyneulink.components.projections.modulatory.controlprojection import ControlProjection
from psyneulink.components.system import System
from psyneulink.library.mechanisms.adaptive import EVCControlMechanism

# Stimulus Mechanisms
Color_Input = TransferMechanism(name='Color Input', function=Linear(slope = 0.2995))
Word_Input = TransferMechanism(name='Word Input', function=Linear(slope = 0.2995))

# Processing Mechanisms (Control)
Color_Hidden = TransferMechanism(name='Colors Hidden',
                               function=Logistic(gain=(1.0, ControlProjection)))
Word_Hidden = TransferMechanism(name='Words Hidden',
                               function=Logistic(gain=(1.0, ControlProjection)))
Output = TransferMechanism(name='Output',
                               function=Logistic(gain=(1.0, ControlProjection)))

# Decision Mechanisms
Decision = DDM(function=BogaczEtAl(drift_rate=(1.0),
                                   threshold=(0.1654),
                                   noise=(0.5),
                                   starting_point=(0),
                                   t0=0.25),
               name='Decision')
# Outcome Mechanisms:
Reward = TransferMechanism(name='Reward')

# Processes:
ColorNamingProcess = Process(
    default_variable=[0],
    pathway=[Color_Input, Color_Hidden, Output, Decision],
    name = 'Color Naming Process')

WordReadingProcess = Process(
    default_variable=[0],
    pathway=[Word_Input, Word_Hidden, Output, Decision],
    name = 'Word Reading Process')

RewardProcess = Process(
    default_variable=[0],
    pathway=[(Reward, 1)],
    name = 'RewardProcess')

# System:
mySystem = System(processes=[ColorNamingProcess, WordReadingProcess, RewardProcess],
                  controller=EVCControlMechanism,
                  enable_controller=True,
                  monitor_for_control=[Reward, (DDM_PROBABILITY_UPPER_THRESHOLD, 1, -1)],
                  name='EVC Gratton System')
# Show characteristics of system:
mySystem.show()
# mySystem.controller.show()
mySystem.show_graph(direction='LR')


stim_list_dict = {Color_Input:[1, 1],
                  Word_Input:[-1, -1],
                  Reward:[36, 29]}

# Run system:
Color_Hidden.reportOutputPref = True
mySystem.reportOutputPref = True
mySystem.controller.reportOutputPref = True
mySystem.run(num_trials=2,
             inputs=stim_list_dict)