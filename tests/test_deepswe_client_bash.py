import hashlib
import pytest
from scripts.patch_deepswe_client_bash import CALL, PATCH, ESCAPE_CALL, ESCAPE_PATCH, patch


def test_pinned_rewrite_preserves_size_and_unrelated_execution_code():
    data=b'permissions-before:'+CALL+b'execution-between:'+ESCAPE_CALL+b'permissions-after'
    result,audit=patch(data,hashlib.sha256(data).hexdigest())
    assert result==b'permissions-before:'+PATCH+b'execution-between:'+ESCAPE_PATCH+b'permissions-after'
    assert len(result)==len(data) and len(audit['sites'])==2


@pytest.mark.parametrize('data,wrong_hash',[(CALL+ESCAPE_CALL,True),(CALL+CALL+ESCAPE_CALL,False),(CALL,False)])
def test_unknown_or_ambiguous_client_is_rejected(data,wrong_hash):
    with pytest.raises(ValueError):
        patch(data,'wrong' if wrong_hash else hashlib.sha256(data).hexdigest())
