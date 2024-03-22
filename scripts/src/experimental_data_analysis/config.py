from scripts.src.core.common.classes import OptionDict
from scripts.src.common.config import Direct as CommonDirect, Keywords
from scripts.src.core.common.config import ParamName
from .inventory import DataModelType, RunningMode


class Direct(object):
    # root_direct = 'scripts'
    # data_direct = '{}/data'.format(root_direct)
    # output_direct = '{}/output'.format(root_direct)
    name = 'experimental_data_analysis'
    output_direct = f'{CommonDirect.output_direct}/{name}'
    common_data_direct = f'{CommonDirect.common_submitted_raw_data_direct}/{name}'


# test_mode = True
# data_model_name = DataModelType.susan_data_neural_cell
data_model_name = DataModelType.fangchao_data_fruit_fly
# data_model_name = DataModelType.fangchao_data_cultured_cell
# data_model_name = DataModelType.yimon_data_cultured_cell
# data_model_name = DataModelType.lianfeng_data_worm
load_previous_results = True


# running_mode = RunningMode.flux_analysis
# running_mode = RunningMode.result_process
running_mode = RunningMode.raw_experimental_data_plotting

# solver_type = ParamName.slsqp_numba_solver
solver_type = ParamName.slsqp_numba_python_solver

loss_type = ParamName.mean_squared_loss

experimental_results_comparison = False
fluxes_comparison = True
output_flux_results = False

loss_percentile = 0.0025
# loss_percentile = None
report_interval = 50
thread_num_constraint = None


def running_settings(test_mode=False):
    parallel_parameter_dict = {
        Keywords.max_optimization_each_generation: 10000,
        Keywords.each_process_optimization_num: 50,
        Keywords.processes_num: 6,
        Keywords.thread_num_constraint: None,
        # Keywords.thread_num_constraint: 4,
        # Keywords.parallel_test: True,
    }
    if test_mode:
        each_case_optimization_num = 10
        parallel_parameter_dict = None
        # parallel_parameter_dict = {
        #     'max_optimization_each_generation': 20,
        #     'each_process_optimization_num': 10,
        #     'processes_num': 1
        # }
    else:
        each_case_optimization_num = 20000
        # parallel_parameter_dict = {
        #     'max_optimization_each_generation': 5000,
        #     'each_process_optimization_num': 50,
        #     # 'processes_num': 4
        #     'processes_num': 6
        # }
        parallel_parameter_dict.update({
            Keywords.max_optimization_each_generation: 5000,
        })
    return each_case_optimization_num, parallel_parameter_dict


class CommonParameters(object):
    common_flux_range = (1, 1000)
    specific_flux_range_dict = {}
    mix_ratio_multiplier = 100
    common_mix_ratio_range = (0.05, 0.95)
    mixed_compartment_list = ('m', 'c')
    model_compartment_set = {'m', 'c', 'e'}
    solver_config_dict = OptionDict({
        ParamName.loss_type: loss_type
    })
    solver_type = ParamName.slsqp_numba_python_solver


