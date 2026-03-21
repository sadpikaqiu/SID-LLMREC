from gnprsid.data.samples import sid_indices_to_token


def test_sid_indices_to_token():
    assert sid_indices_to_token([1, 2, 3]) == "<a_1><b_2><c_3>"
