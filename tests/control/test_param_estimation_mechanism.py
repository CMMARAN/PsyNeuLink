import numpy as np

from psyneulink.components.functions.function import BogaczEtAl, Linear
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.process import Process
from psyneulink.components.projections.modulatory.controlprojection import ControlProjection
from psyneulink.components.system import System
from psyneulink.globals.keywords import ALLOCATION_SAMPLES, IDENTITY_MATRIX, MEAN, RESULT, VARIANCE
from psyneulink.library.mechanisms.processing.integrator.ddm import DDM, DECISION_VARIABLE, PROBABILITY_UPPER_THRESHOLD, \
    RESPONSE_TIME
from psyneulink.library.subsystems.param_estimator.paramestimationcontrolmechanism import \
    ParamEstimationControlMechanism

def test_pecm():

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
    TaskExecutionProcess = Process(
        pathway=[Decision],
        name='TaskExecutionProcess',
    )

    # System:
    mySystem = System(
        processes=[TaskExecutionProcess],
        controller=ParamEstimationControlMechanism(data_in_file='tests/control/hddm_test_data.csv'),
        enable_controller=True,
        monitor_for_control=[
            Decision.PROBABILITY_UPPER_THRESHOLD,
            (Decision.RESPONSE_TIME, -1, 1)],
        name='Param Estimation Test System',
    )

    # TaskExecutionProcess.prefs.paramValidationPref = False
    # RewardProcess.prefs.paramValidationPref = False
    # mySystem.prefs.paramValidationPref = False

    # Stimuli
    stim_list_dict = {
        Decision: [1,1,1,1,1,1,1,1,1,1,1,1,1]
    }

    mySystem.run(inputs=stim_list_dict)