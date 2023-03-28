# Multilingual tasks (DE, FR, ES, IT)
from . import fquad
from . import mlqa
from . import piaf
from . import squad_it
from . import xcopa
from . import xnli
from . import xquad


TASK_REGISTRY = {
    # Multilingual tasks (DE, FR, ES, IT)
    "aam_fquad_fr": fquad.FQuAD,
    "aam_mlqa_de": mlqa.MLQADe,
    "aam_mlqa_en": mlqa.MLQAEn,
    "aam_mlqa_es": mlqa.MLQAEs,
    "aam_piaf_fr": piaf.Piaf,
    "aam_squad_it": squad_it.SQUAD_IT,
    "aam_xcopa_it": xcopa.XCOPA_it,
    "aam_xnli_de": xnli.XNLIDe,
    "aam_xnli_en": xnli.XNLIEn,
    "aam_xnli_es": xnli.XNLIEs,
    "aam_xnli_fr": xnli.XNLIFr,
    "aam_xquad_de": xquad.XQUADDe,
    "aam_xquad_en": xquad.XQUADEn,
    "aam_xquad_es": xquad.XQUADEs,
}
