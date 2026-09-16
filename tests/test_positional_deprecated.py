from lm_eval.utils import positional_deprecated


def test_positional_deprecated_warns_for_plain_function(capsys):
    @positional_deprecated
    def example(value=None):
        return value

    assert example(value="keyword") == "keyword"
    assert capsys.readouterr().out == ""

    assert example("positional") == "positional"
    assert "using example with positional arguments" in capsys.readouterr().out


def test_positional_deprecated_allows_method_receiver(capsys):
    class Example:
        @positional_deprecated
        def method(self, value=None):
            return value

    instance = Example()
    assert instance.method(value="keyword") == "keyword"
    assert capsys.readouterr().out == ""

    assert instance.method("positional") == "positional"
    assert "using method with positional arguments" in capsys.readouterr().out
