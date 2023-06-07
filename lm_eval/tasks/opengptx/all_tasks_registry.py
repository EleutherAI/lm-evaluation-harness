# Added tasks from OpenGPT-X
from . import xnli
from . import xquad
from . import pawsx
from . import x_stance
from . import stereoset
from . import mlqa
from . import gnad10
from . import mlsum
from . import germeval2018
from . import germanquad
from . import germeval2017
from . import german_ler_ppl
from . import german_europarl_ppl
from . import oscar_ppl


TASK_REGISTRY = {
    "ogptx_xstance_de": x_stance.XStanceDE,
    "ogptx_xstance_fr": x_stance.XStanceFR,
    **xquad.construct_tasks(),
    **xnli.construct_tasks(),
    **pawsx.construct_tasks(),
    "ogptx_gnad10": gnad10.GNAD10,
    **stereoset.construct_tasks(),
    **mlqa.construct_tasks(),
    **mlsum.construct_tasks(),
    "ogptx_germeval2018_coarse": germeval2018.GermEval2018,
    "ogptx_germeval2018_fine": germeval2018.GermEval2018_fine,
    "ogptx_germanquad": germanquad.GermanQuAD,
    "ogptx_germeval2017": germeval2017.GermEval2017,
    "ogptx_german_ler_ppl": german_ler_ppl.GermanLERPerplexity,
    "ogptx_german_europarl_ppl": german_europarl_ppl.GermanEuroparlPerplexity,
    "ogptx_oscar_ppl_de": oscar_ppl.OscarPerplexityGerman,
}
