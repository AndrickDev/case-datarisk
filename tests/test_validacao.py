import pytest
from sklearn.model_selection import GroupKFold

from src.config import COL_ID, COL_SAFRA
from src.validacao import split_out_of_time


def test_out_of_time_treina_no_passado_e_testa_nas_k_ultimas(desenvolvimento):
    safras = desenvolvimento[COL_SAFRA].tolist()
    tr, te = split_out_of_time(safras, k=3)
    assert max(safras[i] for i in tr) < min(safras[i] for i in te)
    assert len({safras[i] for i in te}) == 3
    assert len(tr) + len(te) == len(safras)


def test_out_of_time_rejeita_k_maior_que_numero_de_safras():
    with pytest.raises(ValueError):
        split_out_of_time(["2020-01", "2020-02"], k=2)


def test_group_kfold_nao_compartilha_cliente_entre_treino_e_teste(desenvolvimento):
    g = desenvolvimento[COL_ID]
    for tr, te in GroupKFold(5).split(desenvolvimento, groups=g):
        assert set(g.iloc[tr]).isdisjoint(set(g.iloc[te]))
