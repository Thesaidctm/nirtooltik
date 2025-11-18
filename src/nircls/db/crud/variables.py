from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from ..models import Variavel


def get_variavel(session: Session, var_id: int) -> Optional[Variavel]:
    return session.get(Variavel, var_id)


def set_variavel(session: Session, nome: str, var_id: Optional[int] = None) -> Variavel:
    if var_id is not None:
        variavel = session.get(Variavel, var_id)
        if variavel:
            variavel.nome = nome
        else:
            variavel = Variavel(var_id=var_id, nome=nome)
            session.add(variavel)
    else:
        variavel = Variavel(nome=nome)
        session.add(variavel)
    session.commit()
    session.refresh(variavel)
    return variavel
