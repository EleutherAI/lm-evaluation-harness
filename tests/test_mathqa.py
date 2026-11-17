import pytest

from lm_eval.tasks.mathqa.utils import doc_to_choice


# Rows from the regisss/math_qa test split.
@pytest.mark.parametrize(
    "options, expected",
    [
        # row 1
        (
            "a ) 129 , b ) 130 , c ) 124 , d ) 133 , e ) 145",
            ["129", "130", "124", "133", "145"],
        ),
        # row 82: options stored as a list literal
        (
            "['a ) 2 : 3', 'b ) 4 : 5', 'c ) 16 : 9', 'd ) 7 : 9', 'e ) 8 : 5']",
            ["2 : 3", "4 : 5", "16 : 9", "7 : 9", "8 : 5"],
        ),
        # row 895: options that contain " , "
        (
            "a ) 112.5 , 158.5 , b ) 180.5 , 150 , c ) 100 , 200 , d ) 112.5 , 150 , e ) 122.5 , 150",
            ["112.5 , 158.5", "180.5 , 150", "100 , 200", "112.5 , 150", "122.5 , 150"],
        ),
        # row 1701
        (
            "a ) 4 , 3 , b ) 4 , 4 , c ) 3 , 3 , d ) 4 , 5 , e ) 4 , 2",
            ["4 , 3", "4 , 4", "3 , 3", "4 , 5", "4 , 2"],
        ),
    ],
    ids=["plain", "list_literal", "comma_row_895", "comma_row_1701"],
)
def test_mathqa_doc_to_choice(options, expected):
    assert doc_to_choice({"options": options}) == expected
