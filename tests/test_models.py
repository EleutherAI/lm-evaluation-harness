import lm_eval.models as models
import lm_eval.base as base

def test_gpt2():
    gpt2 = models.get_model('gpt2').create_from_arg_string("device=cpu")
    (ll_dog, ig_dog), (ll_cat, ig_cat) = gpt2.loglikelihood([
        ('The quick brown fox jumps over the lazy', ' dog'),
        ('The quick brown fox jumps over the lazy', ' cat'),
    ])

    assert ll_dog > ll_cat
    assert not ig_cat

