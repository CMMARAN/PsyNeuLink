import functools
import numpy as np
import pytest

from psyneulink.components.projections.modulatory.learningprojection import \
    LearningProjection
from psyneulink.components.functions.function import PROB, BogaczEtAl, Reinforcement, SoftMax, Linear, THRESHOLD
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.process import Process
from psyneulink.components.projections.modulatory.controlprojection import ControlProjection
from psyneulink.components.system import System
from psyneulink.globals.keywords import ALLOCATION_SAMPLES, IDENTITY_MATRIX, MEAN, RESULT, VARIANCE, SLOPE, CONTROL
from psyneulink.library.mechanisms.processing.integrator.ddm import DDM, DECISION_VARIABLE, PROBABILITY_UPPER_THRESHOLD, RESPONSE_TIME
from psyneulink.library.subsystems.evc.adaptivepredictionevccontrolmechanism import AdaptivePredictionEVCControlMechanism


def test_reinforcement():
    input_layer = TransferMechanism(
        default_variable=[0, 0, 0],
        name='Input Layer',
    )

    action_selection = TransferMechanism(
        default_variable=[0, 0, 0],
        function=SoftMax(
            output=PROB,
            gain=1.0,
        ),
        name='Action Selection',
    )

    p = Process(
        default_variable=[0, 0, 0],
        size=3,
        pathway=[input_layer, action_selection],
        learning=LearningProjection(learning_function=Reinforcement(learning_rate=0.05)),
        target=0,
    )

    # print ('reward prediction weights: \n', action_selection.input_states[0].path_afferents[0].matrix)
    # print ('targetMechanism weights: \n', action_selection.output_states.sendsToProjections[0].matrix)

    reward_values = [10, 10, 10]

    # Must initialize reward (won't be used, but needed for declaration of lambda function)
    action_selection.output_state.value = [0, 0, 1]
    # Get reward value for selected action)
    reward = lambda: [reward_values[int(np.nonzero(action_selection.output_state.value)[0])]]

    def print_header(system):
        print("\n\n**** TRIAL: ", system.scheduler_processing.clock.simple_time)

    def show_weights():
        print('Reward prediction weights: \n', action_selection.input_states[0].path_afferents[0].mod_matrix)
        print('\nAction selected:  {}; predicted reward: {}'.format(
            np.nonzero(action_selection.output_state.value)[0][0],
            action_selection.output_state.value[np.nonzero(action_selection.output_state.value)[0][0]],
        ))

    input_list = {input_layer: [[1, 1, 1]]}

    s = System(
        processes=[p],
        # learning_rate=0.05,
        targets=[0],
    )

    results = s.run(
        num_trials=10,
        inputs=input_list,
        targets=reward,
        call_before_trial=functools.partial(print_header, s),
        call_after_trial=show_weights,
    )

    results_list = []
    for elem in s.results:
        for nested_elem in elem:
            nested_elem = nested_elem.tolist()
            try:
                iter(nested_elem)
            except TypeError:
                nested_elem = [nested_elem]
            results_list.extend(nested_elem)

    mech_objective_action = s.mechanisms[2]
    mech_learning_input_to_action = s.mechanisms[3]

    reward_prediction_weights = action_selection.input_states[0].path_afferents[0]

    expected_output = [
        (input_layer.output_states.values, [np.array([1., 1., 1.])]),
        (action_selection.output_states.values, [np.array([0., 3.38417298, 0.])]),
        (pytest.helpers.expand_np_ndarray(mech_objective_action.output_states.values), pytest.helpers.expand_np_ndarray([np.array([6.61582702]), np.array(43.7691671006736)])),
        (pytest.helpers.expand_np_ndarray(mech_learning_input_to_action.output_states.values), pytest.helpers.expand_np_ndarray([np.array(
                [0.0, 0.33079135078125005, 0.0, 0.0, 0.33079135078125005, 0.0]
        )])),
        (reward_prediction_weights.mod_matrix, np.array([
            [ 1.,          0.,          0.,        ],
            [ 0.,          3.71496434,  0.,        ],
            [ 0.,          0.,          2.283625,  ],
        ])),
        (results, [
            [np.array([0., 1., 0.])],
            [np.array([0., 1.45, 0.])],
            [np.array([0., 1.8775, 0.])],
            [np.array([0., 2.283625, 0.])],
            [np.array([0., 0., 1.])],
            [np.array([0., 0., 1.45])],
            [np.array([0., 2.66944375, 0.])],
            [np.array([0., 0., 1.8775])],
            [np.array([0., 3.03597156, 0.])],
            [np.array([0., 3.38417298, 0.])]
        ]),
    ]

    for i in range(len(expected_output)):
        val, expected = expected_output[i]
        # setting absolute tolerance to be in accordance with reference_output precision
        # if you do not specify, assert_allclose will use a relative tolerance of 1e-07,
        # which WILL FAIL unless you gather higher precision values to use as reference
        np.testing.assert_allclose(val, expected, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))

def test_learn_over_prediction_process():

    # Mechanisms
    Input = TransferMechanism(name='Input')

    Reward = TransferMechanism(output_states=[RESULT, MEAN, VARIANCE],
                               name='Reward')

    Decision = DDM(
        function=BogaczEtAl(
            drift_rate=(
                1.0,
                ControlProjection(
                    function=Linear,
                    control_signal_params={
                        ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)
                    },
                ),
            ),
            threshold=(
                1.0,
                ControlProjection(
                    function=Linear,
                    control_signal_params={
                        ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)
                    },
                ),
            ),
            noise=(0.5),
            starting_point=(0),
            t0=0.45
        ),
        output_states=[
            DECISION_VARIABLE,
            RESPONSE_TIME,
            PROBABILITY_UPPER_THRESHOLD
        ],
        name='Decision',
    )

    # Processes:
    TaskExecutionProcess = Process(pathway=[Input, IDENTITY_MATRIX, Decision],
                                   name='TaskExecutionProcess')

    RewardProcess = Process(pathway=[Reward],
                            name='RewardProcess')
    myController = AdaptivePredictionEVCControlMechanism()
    # System:
    mySystem = System(
        processes=[TaskExecutionProcess, RewardProcess],
        controller=myController,
        enable_controller=True,
        monitor_for_control=[
            Reward,
            Decision.PROBABILITY_UPPER_THRESHOLD,
            (Decision.RESPONSE_TIME, -1, 1)
        ],
        name='EVC Test System',
    )

    # Stimuli
    stim_list_dict = {
        Input: [0.5, 0.123],
        Reward: [20, 20]
    }

    mySystem.add_prediction_learning([Input, Reward], [0.3481, 0.3481])
    # mySystem.show_graph(show_learning=True)
    print("TARGETS = ", mySystem.targets)
    print(mySystem.controller.prediction_mechanisms)
    target_list_dict = {
        mySystem.controller.prediction_mechanisms[1]: [0.5, 0.123],
        mySystem.controller.prediction_mechanisms[0]: [20, 20]
    }
    input_mechanism_values = []
    # reward_mechanism_values = []
    def check_intermediate_values():
        input_mechanism_values.append(Input.value)
        # reward_mechanism_values.append(Reward.value)
    mySystem.learning = True
    mySystem.run(inputs=stim_list_dict,
                 # targets=target_list_dict,
                 learning=True,
                 call_after_trial=check_intermediate_values)

    RewardPrediction = mySystem.execution_list[3]
    InputPrediction = mySystem.execution_list[4]

    # rearranging mySystem.results into a format that we can compare with pytest
    results_array = []
    for elem in mySystem.results:
        elem_array = []
        for inner_elem in elem:
            elem_array.append(float(inner_elem))
        results_array.append(elem_array)

    # mySystem.results expected output properly formatted
    expected_results_array = [
        [10., 10.0, 0.0, -0.1, 0.48999867, 0.50499983],
        [10., 10.0, 0.0, -0.4, 1.08965888, 0.51998934],
        [10., 10.0, 0.0, 0.7, 2.40680493, 0.53494295],
        [10., 10.0, 0.0, -1., 4.43671978, 0.549834],
        [10., 10.0, 0.0, 0.1, 0.48997868, 0.51998934],
        [10., 10.0, 0.0, -0.4, 1.08459402, 0.57932425],
        [10., 10.0, 0.0, 0.7, 2.36033556, 0.63645254],
        [10., 10.0, 0.0, 1., 4.24948962, 0.68997448],
        [10., 10.0, 0.0, 0.1, 0.48993479, 0.53494295],
        [10., 10.0, 0.0, 0.4, 1.07378304, 0.63645254],
        [10., 10.0, 0.0, 0.7, 2.26686573, 0.72710822],
        [10., 10.0, 0.0, 1., 3.90353015, 0.80218389],
        [10., 10.0, 0.0, 0.1, 0.4898672, 0.549834],
        [10., 10.0, 0.0, -0.4, 1.05791834, 0.68997448],
        [10., 10.0, 0.0, 0.7, 2.14222978, 0.80218389],
        [10., 10.0, 0.0, 1., 3.49637662, 0.88079708],
        [10., 10.0, 0.0, 1., 3.49637662, 0.88079708],
        [15., 15.0, 0.0, 0.1, 0.48999926, 0.50372993],
        [15., 15.0, 0.0, -0.4, 1.08981011, 0.51491557],
        [15., 15.0, 0.0, 0.7, 2.40822035, 0.52608629],
        [15., 15.0, 0.0, 1., 4.44259627, 0.53723096],
        [15., 15.0, 0.0, 0.1, 0.48998813, 0.51491557],
        [15., 15.0, 0.0, 0.4, 1.0869779, 0.55939819],
        [15., 15.0, 0.0, -0.7, 2.38198336, 0.60294711],
        [15., 15.0, 0.0, 1., 4.33535807, 0.64492386],
        [15., 15.0, 0.0, 0.1, 0.48996368, 0.52608629],
        [15., 15.0, 0.0, 0.4, 1.08085171, 0.60294711],
        [15., 15.0, 0.0, 0.7, 2.32712843, 0.67504223],
        [15., 15.0, 0.0, 1., 4.1221271, 0.7396981],
        [15., 15.0, 0.0, 0.1, 0.48992596, 0.53723096],
        [15., 15.0, 0.0, -0.4, 1.07165729, 0.64492386],
        [15., 15.0, 0.0, 0.7, 2.24934228, 0.7396981],
        [15., 15.0, 0.0, 1., 3.84279648, 0.81637827],
        [15., 15.0, 0.0, 1., 3.84279648, 0.81637827]
    ]

    expected_output = [
        # Decision Output | Second Trial
        (Decision.output_states[0].value, np.array(1.0)),

        # Input Prediction Output | Second Trial
        (InputPrediction.output_states[0].value, np.array(0.1865)),

        # RewardPrediction Output | Second Trial
        (RewardPrediction.output_states[0].value, np.array(15.0)),

        # --- Decision Mechanism ---
        #    Output State Values
        #       decision variable
        (Decision.output_states[DECISION_VARIABLE].value, np.array([1.0])),
        #       response time
        (Decision.output_states[RESPONSE_TIME].value, np.array([3.84279648])),
        #       upper bound
        (Decision.output_states[PROBABILITY_UPPER_THRESHOLD].value, np.array([0.81637827])),
        #       lower bound
        # (round(float(Decision.output_states['DDM_probability_lowerBound'].value),3), 0.184),

        # --- Reward Mechanism ---
        #    Output State Values
        #       transfer mean
        (Reward.output_states[RESULT].value, np.array([15.])),
        #       transfer_result
        (Reward.output_states[MEAN].value, np.array(15.0)),
        #       transfer variance
        (Reward.output_states[VARIANCE].value, np.array(0.0)),

        # System Results Array
        #   (all intermediate output values of system)
        (results_array, expected_results_array)
    ]

    for i in range(len(expected_output)):
        val, expected = expected_output[i]
        np.testing.assert_allclose(val, expected, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))

