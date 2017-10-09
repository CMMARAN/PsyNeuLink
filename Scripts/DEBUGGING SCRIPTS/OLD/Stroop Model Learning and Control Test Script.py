from psyneulink.components.functions.function import Linear, Logistic
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.components.system import *
from psyneulink.globals.keywords import *
from psyneulink.library.subsystems.evc.evccontrolmechanism import EVCControlMechanism

process_prefs = {REPORT_OUTPUT_PREF: True,
                 VERBOSE_PREF: False}

system_prefs = {REPORT_OUTPUT_PREF: True,
                VERBOSE_PREF: False}

colors = TransferMechanism(default_variable=[0,0],
                        function=Linear,
                        name="Colors")

words = TransferMechanism(default_variable=[0,0],
                          function=Linear,
                          name="Words")

hidden = TransferMechanism(default_variable=[0,0],
                           function=Logistic(gain=(1.0, CONTROL)),
                           name="Hidden")

response = TransferMechanism(default_variable=[0,0],
                           function=Logistic(),
                           name="Response")

output = TransferMechanism(default_variable=[0,0],
                           function=Logistic,
                           name="Output")

CH_Weights_matrix = np.arange(4).reshape((2,2))
WH_Weights_matrix = np.arange(4).reshape((2,2))
HO_Weights_matrix = np.arange(4).reshape((2,2))

CH_Weights = MappingProjection(name='Color-Hidden Weights',
                        matrix=CH_Weights_matrix
                        )
WH_Weights = MappingProjection(name='Word-Hidden Weights',
                        matrix=WH_Weights_matrix
                        )
HO_Weights = MappingProjection(name='Hidden-Output Weights',
                        matrix=HO_Weights_matrix
                        )

color_naming_process = Process(
    default_variable=[1, 2.5],
    pathway=[colors, CH_Weights, hidden, HO_Weights, response],
    learning=LEARNING,
    target=[2,2],
    name='Color Naming',
    prefs=process_prefs)

word_reading_process = Process(
    default_variable=[.5, 3],
    pathway=[words, WH_Weights, hidden],
    name='Word Reading',
    learning=LEARNING,
    target=[3,3],
    prefs=process_prefs)

color_naming_process.execute()
word_reading_process.execute()

mySystem = System(processes=[color_naming_process, word_reading_process],
                  targets=[0,0],
                  controller=EVCControlMechanism,
                  enable_controller=True,
                  name='Stroop Model',
                  prefs=system_prefs)

# mySystem.show_graph_with_learning()
mySystem.show_graph_with_control()

def print_header():
    print("\n\n**** TRIAL: ", CentralClock.trial)

def show_target():
    print ('\nColor Naming\n\tInput: {}\n\tTarget: {}'.
           # format(color_naming_process.input, color_naming_process.target))
           format(colors.input_values, mySystem.targets))
    print ('Wording Reading:\n\tInput: {}\n\tTarget: {}\n'.
           # format(word_reading_process.input, word_reading_process.target))
           format(words.input_values, mySystem.targets))
    print ('Response: \n', response.output_values[0])
    print ('Hidden-Output:')
    print (HO_Weights.matrix)
    print ('Color-Hidden:')
    print (CH_Weights.matrix)
    print ('Word-Hidden:')
    print (WH_Weights.matrix)


stim_list_dict = {colors:[[1, 1]],
                  words:[[-2, -2]]}

target_list_dict = {response:[[1, 1]]}

mySystem.run(num_trials=2,
            inputs=stim_list_dict,
            targets=target_list_dict,
            call_before_trial=print_header,
            call_after_trial=show_target)