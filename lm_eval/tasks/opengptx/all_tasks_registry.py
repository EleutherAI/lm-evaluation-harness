# OpenGPT-X tasks
from . import flores200
from . import german_europarl_ppl
from . import german_ler_ppl
from . import germanquad
from . import germeval2017
from . import germeval2018
from . import gnad10
from . import mlqa
from . import mlsum
from . import oscar_ppl
from . import pawsx
from . import stereoset
from . import wino_x
from . import xcsr
from . import xlwic
from . import x_stance
from . import xquad
from . import xnli

########################################
# Translation tasks
########################################

TASK_REGISTRY_TMP = {
    # OpenGPT-X tasks
    "german_europarl_ppl": german_europarl_ppl.GermanEuroparlPerplexity,
    "german_ler_ppl": german_ler_ppl.GermanLERPerplexity,
    "germanquad": germanquad.GermanQuAD,
    "germeval2017": germeval2017.GermEval2017,
    "germeval2018_coarse": germeval2018.GermEval2018,
    "germeval2018_fine": germeval2018.GermEval2018_fine,
    "gnad10": gnad10.GNAD10,
    **mlqa.construct_tasks(),
    **mlsum.construct_tasks(),
    "oscar_ppl_de": oscar_ppl.OscarPerplexityGerman,
    **pawsx.construct_tasks(),
    **stereoset.construct_tasks(),
    **xcsr.construct_tasks(),
    "wino_de": wino_x.WinograndeXDe,
    "xlwic_de": xlwic.WordsInContextDe,
    "xlwic_it": xlwic.WordsInContextIt,
    "xstance_de": x_stance.XStanceDE,
    "xstance_fr": x_stance.XStanceFR,
    **xquad.construct_tasks(),
    **xnli.construct_tasks(),
    **flores200.construct_lang_tasks(),
    **flores200.construct_trans_tasks(),
}

# add a prefix to tasks implemented by OpenGPT-X
PREFIX = "ogx_"
TASK_REGISTRY = {}
for task in TASK_REGISTRY_TMP:
    TASK_REGISTRY[PREFIX + task] = TASK_REGISTRY_TMP[task]
