from .base_model import reaction_dict, emu_excluded_metabolite_set, symmetrical_metabolite_set, \
    added_input_metabolite_set, model_compartment_set, composite_reaction_list, \
    ModelKeyword

reaction_dict['tca_reaction'].append(
    {
        'id': 'GLC_supplement',
        'sub': [('GLC_stock', 'abcdef')],
        'pro': [('GLC_c', 'abcdef')],
        'reverse': True
    }
)

# reaction_dict['tca_reaction'].append(
#     {
#         'id': 'CIT_supplement',
#         'sub': [('CIT_stock', 'abcdef')],
#         'pro': [('CIT_m', 'abcdef')],
#         'reverse': True
#     }
# )

emu_excluded_metabolite_set |= {'CIT_stock', 'GLC_stock'}

balance_excluded_metabolite_set = emu_excluded_metabolite_set

if composite_reaction_list is None:
    composite_reaction_list = []
composite_reaction_list.extend([
    {
        'id': 'GLC_supplement_net',
        'comp': [('GLC_supplement', ), ('GLC_supplement__R', -1)],
        ModelKeyword.flux_range: (-1000, 1000)
    },
    # {
    #     'id': 'CIT_supplement_net',
    #     'comp': [('CIT_supplement', ), ('CIT_supplement__R', -1)],
    #     ModelKeyword.flux_range: (-1000, 1000)
    # }
])

