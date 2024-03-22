from .base_model import reaction_dict, emu_excluded_metabolite_set, symmetrical_metabolite_set, \
    added_input_metabolite_set, model_compartment_set, composite_reaction_list, \
    ModelKeyword

reaction_dict['intake_secret_reaction'].append(
    {
        'id': 'GLC_unlabelled_input',
        'sub': [('GLC_unlabelled_e', 'abcdef')],
        'pro': [('GLC_c', 'abcdef')],
    },
)

emu_excluded_metabolite_set |= {'GLC_unlabelled_e'}

balance_excluded_metabolite_set = emu_excluded_metabolite_set

if composite_reaction_list is None:
    composite_reaction_list = []
composite_reaction_list.extend([
    {
        'id': 'GLC_total_input',
        'comp': [('GLC_input', ), ('GLC_unlabelled_input', )],
        ModelKeyword.flux_range: ModelKeyword.add_range_type
    },
])

