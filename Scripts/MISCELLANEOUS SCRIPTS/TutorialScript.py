from psyneulink.components.mechanisms.processing.transfermechanism import *
from psyneulink.components.process import *

# simple_ddm_process = Process('Simple DDM Process')
# simple_ddm_process.execute([1])

my_transfer_mechanism = TransferMechanism(name="My TransferMechanism Mechanism",
                                 params={FUNCTION:Logistic,
                                         FUNCTION_PARAMS:{
                                             kwTransfer_Gain:5,
                                             kwTransfer_Bias:0
                                         }})

# simple_transfer_process = Process(name='Simple TransferMechanism Process',
#                                   params={PATHWAY:[TransferMechanism]})
# simple_transfer_process.execute([0.5])

simple_transfer_process = Process(name='Simple TransferMechanism Process',
                                  params={PATHWAY:[my_transfer_mechanism]})
simple_transfer_process.execute([1])