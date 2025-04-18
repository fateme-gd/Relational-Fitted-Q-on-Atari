import os

from blendrl.nsfr.nsfr.facts_converter import FactsConverter
from blendrl.nsfr.nsfr.utils.logic import get_lang, get_blender_lang, build_infer_module
from blendrl.nsfr.nsfr.nsfr import NSFReasoner
from blendrl.nsfr.nsfr.valuation import ValuationModule


def get_nsfr_model(env_name: str, rules: str, device: str, train=False, explain=False):
    current_path = os.path.dirname(__file__)
    lark_path = os.path.join(current_path, 'lark/exp.lark')
    lang_base_path = f"ins/envs/{env_name}/logic/"

    lang, clauses, bk, atoms = get_lang(lark_path, lang_base_path, rules)

    val_fn_path = f"ins/envs/{env_name}/valuation.py"
    val_module = ValuationModule(val_fn_path, lang, device)

    FC = FactsConverter(lang=lang, valuation_module=val_module, device=device)
    prednames = []
    for clause in clauses:
        if clause.head.pred.name not in prednames:
            prednames.append(clause.head.pred.name)
    m = len(prednames)
    # m = 5
    IM = build_infer_module(clauses, atoms, lang, m=m, infer_step=2, train=train, device=device)
    # Neuro-Symbolic Forward Reasoner
    NSFR = NSFReasoner(facts_converter=FC, infer_module=IM, atoms=atoms, bk=bk, clauses=clauses, device=device, train=train, explain=explain)
    return NSFR


def get_blender_nsfr_model(env_name: str, rules: str, device: str, train=False, mode='normal', explain=False):
    current_path = os.path.dirname(__file__)
    lark_path = os.path.join(current_path, 'lark/exp.lark')
    lang_base_path = f"ins/envs/{env_name}/logic/"

    lang, clauses, bk, atoms = get_blender_lang(lark_path, lang_base_path, rules)

    val_fn_path = f"ins/envs/{env_name}/valuation.py"
    val_module = ValuationModule(val_fn_path, lang, device)

    FC = FactsConverter(lang=lang, valuation_module=val_module, device=device)
    prednames = []
    for clause in clauses:
        if clause.head.pred.name not in prednames:
            prednames.append(clause.head.pred.name)
    # if train:
    #     m = len(prednames)
    # else:
    #     m = len(clauses)
    m = len(clauses)
    # m = 5
    IM = build_infer_module(clauses, atoms, lang, m=m, infer_step=2, train=train, device=device)
    # Neuro-Symbolic Forward Reasoner
    NSFR = NSFReasoner(facts_converter=FC, infer_module=IM, atoms=atoms, bk=bk, clauses=clauses, device=device, train=train, explain=explain)
    return NSFR