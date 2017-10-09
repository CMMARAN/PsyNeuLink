from psyneulink.components.functions.function import Linear, Logistic
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.system import *
from psyneulink.globals.keywords import *

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

response = TransferMechanism(default_variable=[0,0],
                           function=Logistic,
                           name="Response")

color_naming_process = Process(
    default_variable=[1, 2.5],
    # pathway=[(colors, 0), FULL_CONNECTIVITY_MATRIX, (response,0)],
    pathway=[colors, FULL_CONNECTIVITY_MATRIX, response],
    learning=LEARNING_PROJECTION,
    target=[1,2],
    name='Color Naming',
    prefs=process_prefs
)

word_reading_process = Process(
    default_variable=[.5, 3],
    pathway=[words, FULL_CONNECTIVITY_MATRIX, response],
    name='Word Reading',
    learning=LEARNING_PROJECTION,
    target=[3,4],
    prefs=process_prefs
)

mySystem = System(processes=[color_naming_process, word_reading_process],
                  name='Stroop Model',
                  targets=[0,0],
                  prefs=system_prefs,
                  )

# TEST REPORT_OUTPUT_PREFs:
# colors.reportOutputPref = True
# words.reportOutputPref = True
# response.reportOutputPref = True
# color_naming_process.reportOutputPref = False
# word_reading_process.reportOutputPref =  False
# process_prefs.reportOutputPref = PreferenceEntry(True, PreferenceLevel.CATEGORY)

# mySystem.reportOutputPref = True

# # Execute processes:
# for i in range(10):
#     color_naming_process.execute(input=[1, 1],target=[0,1])
#     print(response.input_state.path_afferents[0].matrix)
#     print(response.input_state.path_afferents[1].matrix)
#
#     word_reading_process.execute(input=[1, 1], target=[1,0])
#     print(response.input_state.path_afferents[0].matrix)
#     print(response.input_state.path_afferents[1].matrix)

# Execute system:
# mySystem.execute(input=[[1,1],[1,1]])

# mySystem.show_graph()

stim_dict = {colors:[[1,0],[0,1]],
             words:[[0,1],[1,0]]}
target_dict= {response:[[1,0],[0,1]]}

mySystem.run(num_trials=2,
             inputs=stim_dict,
             targets=target_dict)


# SHOW OPTIONS:
# mySystem.show()
# mySystem.controller.show()
