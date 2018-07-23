from psyneulink.components.functions.function import Linear, SimpleIntegrator, ModulationParam
from psyneulink.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.library.mechanisms.processing.transfer.recurrenttransfermechanism import RecurrentTransferMechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.components.projections.modulatory.controlprojection import ControlProjection
from psyneulink.library.subsystems.agt.lccontrolmechanism import LCControlMechanism
from psyneulink.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.compositions.composition import Composition
from psyneulink.scheduling.scheduler import Scheduler
from psyneulink import SLOPE

from itertools import product
import numpy as np
import pytest

@pytest.mark.composition
@pytest.mark.benchmark(group="LinearComposition")
@pytest.mark.parametrize("llvm", ['Python', 'LLVM'])
def test_run_composition(benchmark, llvm):
    comp = Composition()
    A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
    B = TransferMechanism(function=Linear(slope=5.0))
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
    comp._analyze_graph()
    sched = Scheduler(composition=comp)
    output = benchmark(comp.run, inputs={A: [[1.0]]}, scheduler_processing=sched, bin_execute=(llvm == 'LLVM'))
    assert 25 == output[0][0]


@pytest.mark.skip
@pytest.mark.composition
@pytest.mark.benchmark(group="LinearComposition")
@pytest.mark.parametrize("llvm", ['Python', 'LLVM'])
def test_run_composition_default(benchmark, llvm):
    comp = Composition()
    A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
    B = TransferMechanism(function=Linear(slope=5.0))
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
    comp._analyze_graph()
    sched = Scheduler(composition=comp)
    output = benchmark(comp.run, scheduler_processing=sched, bin_execute=(llvm == 'LLVM'))
    assert 25 == output[0][0]


@pytest.mark.composition
@pytest.mark.benchmark(group="LinearComposition Pathway 5")
@pytest.mark.parametrize("llvm", ['Python', 'LLVM'])
def test_LPP(benchmark, llvm):

    var = 1.0
    comp = Composition()
    A = TransferMechanism(default_variable=var, name="A", function=Linear(slope=2.0))   # 1 x 2 = 2
    B = TransferMechanism(default_variable=var, name="B", function=Linear(slope=2.0))   # 2 x 2 = 4
    C = TransferMechanism(default_variable=var, name="C", function=Linear(slope=2.0))   # 4 x 2 = 8
    D = TransferMechanism(default_variable=var, name="D", function=Linear(slope=2.0))   # 8 x 2 = 16
    E = TransferMechanism(default_variable=var, name="E", function=Linear(slope=2.0))  # 16 x 2 = 32
    comp.add_linear_processing_pathway([A, B, C, D, E])
    comp._analyze_graph()
    inputs_dict = {A: [var]}
    sched = Scheduler(composition=comp)
    output = benchmark(comp.execute, inputs=inputs_dict, scheduler_processing=sched, bin_execute=(llvm=='LLVM'))
    assert 32 == output[0][0]


@pytest.mark.composition
@pytest.mark.benchmark(group="LinearComposition Vector")
@pytest.mark.parametrize("llvm, vector_length", product(('Python', 'LLVM'), [2**x for x in range(1)]))
def test_run_composition_vector(benchmark, llvm, vector_length):
    var = [1.0 for x in range(vector_length)];
    comp = Composition()
    A = IntegratorMechanism(default_variable=var, function=Linear(slope=5.0))
    B = TransferMechanism(default_variable=var, function=Linear(slope=5.0))
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
    comp._analyze_graph()
    sched = Scheduler(composition=comp)
    output = benchmark(comp.run, inputs={A: [var]}, scheduler_processing=sched, bin_execute=(llvm=='LLVM'))
    assert np.allclose([25.0 for x in range(vector_length)], output[0])


@pytest.mark.composition
@pytest.mark.benchmark(group="Merge composition scalar")
@pytest.mark.parametrize("mode", ['Python', 'LLVM'])
def test_5_mechanisms_2_origins_1_terminal(benchmark, mode):
    # A ----> C --
    #              ==> E
    # B ----> D --

    # 5 x 1 = 5 ----> 5 x 5 = 25 --
    #                                25 + 25 = 50  ==> 50 * 5 = 250
    # 5 x 1 = 5 ----> 5 x 5 = 25 --

    comp = Composition()
    A = TransferMechanism(name="A", function=Linear(slope=1.0))
    B = TransferMechanism(name="B", function=Linear(slope=1.0))
    C = TransferMechanism(name="C", function=Linear(slope=5.0))
    D = TransferMechanism(name="D", function=Linear(slope=5.0))
    E = TransferMechanism(name="E", function=Linear(slope=5.0))
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_mechanism(C)
    comp.add_mechanism(D)
    comp.add_projection(A, MappingProjection(sender=A, receiver=C), C)
    comp.add_projection(B, MappingProjection(sender=B, receiver=D), D)
    comp.add_mechanism(E)
    comp.add_projection(C, MappingProjection(sender=C, receiver=E), E)
    comp.add_projection(D, MappingProjection(sender=D, receiver=E), E)
    comp._analyze_graph()
    inputs_dict = {A: [5.0],
                   B: [5.0]}
    sched = Scheduler(composition=comp)
    output = benchmark(comp.run, inputs=inputs_dict, scheduler_processing=sched, bin_execute=(mode=='LLVM'))
    assert 250 == output[0][0]


@pytest.mark.composition
@pytest.mark.benchmark(group="Control composition scalar")
@pytest.mark.parametrize("mode", ['Python', 'LLVM'])
def test_3_mechanisms_2_origins_1_multi_control_1_terminal(benchmark, mode):
    #
    #   B--A
    #  /    \
    # C------D
    #  \     |
    #   -----+-> E
    #
    # C: 4 x 5 = 20
    # B: 20 x 1 = 20
    # A: f(20)[0] = 0.50838675
    # D: 20 x 5 x 0.50838675 = 50.83865743
    # E: (20 + 50.83865743) x 5 = 354.19328716

    comp = Composition()
    C = TransferMechanism(name="C", function=Linear(slope=5.0))
    D = TransferMechanism(name="D", function=Linear(slope=5.0))
    B = ObjectiveMechanism(function=Linear,
                           #monitored_output_states=[C], #setup later
                           name='LC ObjectiveMechanism')
    A = LCControlMechanism(name="A")
#                           modulated_mechanisms=D,
#                           objective_mechanism=B # setup later
    E = TransferMechanism(name="E", function=Linear(slope=5.0))
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_mechanism(C)
    comp.add_mechanism(D)
    comp.add_mechanism(E)

    comp.add_projection(A, ControlProjection(sender=A.control_signals[0], receiver=D.parameter_states[SLOPE]), D)
    comp.add_projection(C, MappingProjection(sender=C, receiver=B), B)
    comp.add_projection(C, MappingProjection(sender=C, receiver=D), D)
    comp.add_projection(C, MappingProjection(sender=C, receiver=E), E)
    comp.add_projection(D, MappingProjection(sender=D, receiver=E), E)
    comp.add_projection(B, MappingProjection(sender=B, receiver=A), A)
    comp._analyze_graph()

    inputs_dict = {C: [4.0]}
    sched = Scheduler(composition=comp)
    output = comp.run(inputs=inputs_dict, scheduler_processing=sched, bin_execute=(mode=='LLVM'))
    assert np.allclose(output, 354.19328716)
    benchmark(comp.run, inputs=inputs_dict, scheduler_processing=sched, bin_execute=(mode=='LLVM'))


@pytest.mark.composition
@pytest.mark.benchmark(group="Control composition scalar")
@pytest.mark.parametrize("mode", ['Python', 'LLVM'])
def test_3_mechanisms_2_origins_1_additive_control_1_terminal(benchmark, mode):
    #
    #   B--A
    #  /    \
    # C------D
    #  \     |
    #   -----+-> E
    #
    # C: 4 x 5 = 20
    # B: 20 x 1 = 20
    # A: f(20)[0] = 0.50838675
    # D: 20 x 5 + 0.50838675 = 100.50838675
    # E: (20 + 100.50838675) x 5 = 650.83865743

    comp = Composition()
    C = TransferMechanism(name="C", function=Linear(slope=5.0))
    D = TransferMechanism(name="D", function=Linear(slope=5.0))
    B = ObjectiveMechanism(function=Linear,
                           #monitored_output_states=[C], #setup later
                           name='LC ObjectiveMechanism')
    A = LCControlMechanism(name="A", modulation=ModulationParam.ADDITIVE)
#                           modulated_mechanisms=D,
#                           objective_mechanism=B # setup later
    E = TransferMechanism(name="E", function=Linear(slope=5.0))
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_mechanism(C)
    comp.add_mechanism(D)
    comp.add_mechanism(E)

    comp.add_projection(A, ControlProjection(sender=A.control_signals[0], receiver=D.parameter_states[SLOPE]), D)
    comp.add_projection(C, MappingProjection(sender=C, receiver=B), B)
    comp.add_projection(C, MappingProjection(sender=C, receiver=D), D)
    comp.add_projection(C, MappingProjection(sender=C, receiver=E), E)
    comp.add_projection(D, MappingProjection(sender=D, receiver=E), E)
    comp.add_projection(B, MappingProjection(sender=B, receiver=A), A)
    comp._analyze_graph()

    inputs_dict = {C: [4.0]}
    sched = Scheduler(composition=comp)
    output = comp.run(inputs=inputs_dict, scheduler_processing=sched, bin_execute=(mode=='LLVM'))
    assert np.allclose(output, 650.83865743)
    benchmark(comp.run, inputs=inputs_dict, scheduler_processing=sched, bin_execute=(mode=='LLVM'))


@pytest.mark.composition
@pytest.mark.benchmark(group="Control composition scalar")
@pytest.mark.parametrize("mode", ['Python', 'LLVM'])
def test_3_mechanisms_2_origins_1_override_control_1_terminal(benchmark, mode):
    #
    #   B--A
    #  /    \
    # C------D
    #  \     |
    #   -----+-> E
    #
    # C: 4 x 5 = 20
    # B: 20 x 1 = 20
    # A: f(20)[0] = 0.50838675
    # D: 20 x 0.50838675 = 10.167735
    # E: (20 + 10.167735) x 5 = 150.83865743

    comp = Composition()
    C = TransferMechanism(name="C", function=Linear(slope=5.0))
    D = TransferMechanism(name="D", function=Linear(slope=5.0))
    B = ObjectiveMechanism(function=Linear,
                           #monitored_output_states=[C], #setup later
                           name='LC ObjectiveMechanism')
    A = LCControlMechanism(name="A", modulation=ModulationParam.OVERRIDE)
#                           modulated_mechanisms=D,
#                           objective_mechanism=B # setup later
    E = TransferMechanism(name="E", function=Linear(slope=5.0))
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_mechanism(C)
    comp.add_mechanism(D)
    comp.add_mechanism(E)

    comp.add_projection(A, ControlProjection(sender=A.control_signals[0], receiver=D.parameter_states[SLOPE]), D)
    comp.add_projection(C, MappingProjection(sender=C, receiver=B), B)
    comp.add_projection(C, MappingProjection(sender=C, receiver=D), D)
    comp.add_projection(C, MappingProjection(sender=C, receiver=E), E)
    comp.add_projection(D, MappingProjection(sender=D, receiver=E), E)
    comp.add_projection(B, MappingProjection(sender=B, receiver=A), A)
    comp._analyze_graph()

    inputs_dict = {C: [4.0]}
    sched = Scheduler(composition=comp)
    output = comp.run(inputs=inputs_dict, scheduler_processing=sched, bin_execute=(mode=='LLVM'))
    assert np.allclose(output, 150.83865743)
    benchmark(comp.run, inputs=inputs_dict, scheduler_processing=sched, bin_execute=(mode=='LLVM'))

@pytest.mark.composition
@pytest.mark.benchmark(group="Control composition scalar")
@pytest.mark.parametrize("mode", ['Python', 'LLVM'])
def test_3_mechanisms_2_origins_1_disable_control_1_terminal(benchmark, mode):
    #
    #   B--A
    #  /    \
    # C------D
    #  \     |
    #   -----+-> E
    #
    # C: 4 x 5 = 20
    # B: 20 x 1 = 20
    # A: f(20)[0] = 0.50838675
    # D: 20 x 5 = 100
    # E: (20 + 100) x 5 = 600

    comp = Composition()
    C = TransferMechanism(name="C", function=Linear(slope=5.0))
    D = TransferMechanism(name="D", function=Linear(slope=5.0))
    B = ObjectiveMechanism(function=Linear,
                           #monitored_output_states=[C], #setup later
                           name='LC ObjectiveMechanism')
    A = LCControlMechanism(name="A", modulation=ModulationParam.DISABLE)
#                           modulated_mechanisms=D,
#                           objective_mechanism=B # setup later
    E = TransferMechanism(name="E", function=Linear(slope=5.0))
    comp.add_mechanism(A)
    comp.add_mechanism(B)
    comp.add_mechanism(C)
    comp.add_mechanism(D)
    comp.add_mechanism(E)

    comp.add_projection(A, ControlProjection(sender=A.control_signals[0], receiver=D.parameter_states[SLOPE]), D)
    comp.add_projection(C, MappingProjection(sender=C, receiver=B), B)
    comp.add_projection(C, MappingProjection(sender=C, receiver=D), D)
    comp.add_projection(C, MappingProjection(sender=C, receiver=E), E)
    comp.add_projection(D, MappingProjection(sender=D, receiver=E), E)
    comp.add_projection(B, MappingProjection(sender=B, receiver=A), A)
    comp._analyze_graph()

    inputs_dict = {C: [4.0]}
    sched = Scheduler(composition=comp)
    output = comp.run(inputs=inputs_dict, scheduler_processing=sched, bin_execute=(mode=='LLVM'))
    assert np.allclose(output, 600)
    benchmark(comp.run, inputs=inputs_dict, scheduler_processing=sched, bin_execute=(mode=='LLVM'))


@pytest.mark.composition
@pytest.mark.benchmark(group="Merge composition scalar")
@pytest.mark.parametrize("mode", ['Python', 'LLVM'])
def test_3_mechanisms_2_origins_1_terminal(benchmark, mode):
    # C --
    #              ==> E
    # D --

    # 5 x 5 = 25 --
    #                25 + 25 = 50  ==> 50 * 5 = 250
    # 5 x 5 = 25 --

    comp = Composition()
    C = TransferMechanism(name="C", function=Linear(slope=5.0))
    D = TransferMechanism(name="D", function=Linear(slope=5.0))
    E = TransferMechanism(name="E", function=Linear(slope=5.0))
    comp.add_mechanism(C)
    comp.add_mechanism(D)
    comp.add_mechanism(E)
    comp.add_projection(C, MappingProjection(sender=C, receiver=E), E)
    comp.add_projection(D, MappingProjection(sender=D, receiver=E), E)
    comp._analyze_graph()
    inputs_dict = {C: [5.0],
                   D: [5.0]}
    sched = Scheduler(composition=comp)
    output = benchmark(comp.run, inputs=inputs_dict, scheduler_processing=sched, bin_execute=(mode=='LLVM'))
    assert 250 == output[0][0]


@pytest.mark.composition
@pytest.mark.benchmark(group="Merge composition scalar MIMO")
@pytest.mark.parametrize("mode", ['Python', 'LLVM'])
def test_3_mechanisms_2_origins_1_terminal_mimo_last(benchmark, mode):
    # C --
    #              ==> E
    # D --

    # [6] x 5 = [30] --
    #                            [30, 40] * 5 = [150, 200]
    # [8] x 5 = [40] --

    comp = Composition()
    C = TransferMechanism(name="C", function=Linear(slope=5.0))
    D = TransferMechanism(name="D", function=Linear(slope=5.0))
    E = TransferMechanism(name="E", input_states=['a', 'b'], function=Linear(slope=5.0))
    comp.add_mechanism(C)
    comp.add_mechanism(D)
    comp.add_mechanism(E)
    comp.add_projection(C, MappingProjection(sender=C, receiver=E.input_states['a']), E)
    comp.add_projection(D, MappingProjection(sender=D, receiver=E.input_states['b']), E)
    comp._analyze_graph()
    inputs_dict = {C: [6.0],
                   D: [8.0]}
    sched = Scheduler(composition=comp)
    output = benchmark(comp.run, inputs=inputs_dict, scheduler_processing=sched, bin_execute=(mode=='LLVM'))
    assert np.allclose([[150], [200]], output)


@pytest.mark.composition
@pytest.mark.benchmark(group="Merge composition scalar MIMO")
@pytest.mark.parametrize("mode", ['Python', 'LLVM'])
def test_3_mechanisms_2_origins_1_terminal_mimo_parallel(benchmark, mode):
    # C --
    #              ==> E
    # D --

    # [5, 6] x 5 = [25, 30] --
    #                            [25 + 35, 30 + 40] = [60, 70]  ==> [60, 70] * 5 = [300, 350]
    # [7, 8] x 5 = [35, 40] --

    comp = Composition()
    C = TransferMechanism(name="C", input_states=['a', 'b'], function=Linear(slope=5.0))
    D = TransferMechanism(name="D", input_states=['a', 'b'], function=Linear(slope=5.0))
    E = TransferMechanism(name="E", input_states=['a', 'b'], function=Linear(slope=5.0))
    comp.add_mechanism(C)
    comp.add_mechanism(D)
    comp.add_mechanism(E)
    comp.add_projection(C, MappingProjection(sender=C.output_states[0], receiver=E.input_states['a']), E)
    comp.add_projection(C, MappingProjection(sender=C.output_states[1], receiver=E.input_states['b']), E)
    comp.add_projection(D, MappingProjection(sender=D.output_states[0], receiver=E.input_states['a']), E)
    comp.add_projection(D, MappingProjection(sender=D.output_states[1], receiver=E.input_states['b']), E)
    comp._analyze_graph()
    inputs_dict = {C: [[5.0], [6.0]],
                   D: [[7.0], [8.0]]}
    sched = Scheduler(composition=comp)
    output = benchmark(comp.run, inputs=inputs_dict, scheduler_processing=sched, bin_execute=(mode=='LLVM'))
    assert np.allclose([[300], [350]], output)


@pytest.mark.composition
@pytest.mark.benchmark(group="Merge composition scalar MIMO")
@pytest.mark.parametrize("mode", ['Python', 'LLVM'])
def test_3_mechanisms_2_origins_1_terminal_mimo_all_sum(benchmark, mode):
    # C --
    #              ==> E
    # D --

    # [5, 6] x 5 = [25, 30] --
    #                            [25 + 35 + 30 + 40] = 130  ==> 130 * 5 = 650
    # [7, 8] x 5 = [35, 40] --

    comp = Composition()
    C = TransferMechanism(name="C", input_states=['a', 'b'], function=Linear(slope=5.0))
    D = TransferMechanism(name="D", input_states=['a', 'b'], function=Linear(slope=5.0))
    E = TransferMechanism(name="E", function=Linear(slope=5.0))
    comp.add_mechanism(C)
    comp.add_mechanism(D)
    comp.add_mechanism(E)
    comp.add_projection(C, MappingProjection(sender=C.output_states[0], receiver=E), E)
    comp.add_projection(C, MappingProjection(sender=C.output_states[1], receiver=E), E)
    comp.add_projection(D, MappingProjection(sender=D.output_states[0], receiver=E), E)
    comp.add_projection(D, MappingProjection(sender=D.output_states[1], receiver=E), E)
    comp._analyze_graph()
    inputs_dict = {C: [[5.0], [6.0]],
                   D: [[7.0], [8.0]]}
    sched = Scheduler(composition=comp)
    output = benchmark(comp.run, inputs=inputs_dict, scheduler_processing=sched, bin_execute=(mode=='LLVM'))
    assert np.allclose([[650]], output)

@pytest.mark.composition
@pytest.mark.benchmark(group="Recurrent")
@pytest.mark.parametrize("mode", ['Python', 'LLVM'])
def test_run_recurrent_transfer_mechanism(benchmark, mode):
    comp = Composition()
    A = RecurrentTransferMechanism(size=3, function=Linear(slope=5.0), name="A")
    comp.add_mechanism(A)
    comp._analyze_graph()
    sched = Scheduler(composition=comp)
    output1 = comp.run(inputs={A: [[1.0, 2.0, 3.0]]}, scheduler_processing=sched, bin_execute=(mode == 'LLVM'))
    assert np.allclose([5.0, 10.0, 15.0], output1)
    output2 = comp.run(inputs={A: [[1.0, 2.0, 3.0]]}, scheduler_processing=sched, bin_execute=(mode == 'LLVM'))
    # Using the hollow matrix: (10 + 15 + 1) * 5 = 130,
    #                          ( 5 + 15 + 2) * 5 = 110,
    #                          ( 5 + 10 + 3) * 5 = 90
    assert np.allclose([130.0, 110.0, 90.0], output2)
    benchmark(comp.run, inputs={A: [[1.0, 2.0, 3.0]]}, scheduler_processing=sched, bin_execute=(mode == 'LLVM'))
