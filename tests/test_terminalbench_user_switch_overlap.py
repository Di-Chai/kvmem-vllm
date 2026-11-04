import pytest

from scripts import continue_terminalbench_user_switch as overlap
from scripts import run_terminalbench_user_switch as switch
from test_terminalbench_network_audit_recovery import campaign as network_campaign, read, sha, write
from test_terminalbench_setup_audit_recovery import campaign as setup_campaign
from test_terminalbench_recorder_audit_recovery import campaign as recorder_campaign
from test_terminalbench_query_audit_recovery import campaign as query_campaign
from test_terminalbench_user_switch import campaign as stop_campaign


@pytest.fixture
def campaign(stop_campaign):
    c=stop_campaign;t=c['transition']
    audited=switch.audit(c['pair'],t,live_probe=lambda _:False)
    write(t/'audit.json',audited)
    write(t/'finalizer-adoption.json',dict(previous_controller=dict(pid=999,created=1),
        finalizer=dict(pid=998,created=1),audit_sha256=sha(t/'audit.json'),
        authorization_sha256=sha(t/'authorization.json')))
    c['audited']=audited
    return c


def test_retained_cpu_review_can_overlap_lru_startup_after_integrity_audit(campaign):
    c=campaign;_,audited=overlap.validate_adoption(c['pair'],c['transition'],live_probe=lambda _:False)
    assert audited['summary']['scored']==82 and audited['summary']['cancelled_by_user']==3


@pytest.mark.parametrize('damage',['live_predecessor','changed_score','lru_already_started'])
def test_adoption_never_duplicates_a_suite_or_accepts_changed_scores(campaign,damage):
    c=campaign
    if damage=='changed_score':
        p=c['arm']/'jobs/formal'/c['trials'][8]/'result.json';p.write_text(p.read_text()+' ')
    elif damage=='lru_already_started':(c['lru']/'started').write_text('started')
    with pytest.raises(ValueError):
        overlap.validate_adoption(c['pair'],c['transition'],live_probe=lambda _:damage=='live_predecessor')


@pytest.mark.parametrize('scored',[82,85])
def test_final_review_must_preserve_missing_rewards(campaign,scored):
    c=campaign;arm=c['arm'];measurement=dict(passed=True)
    write(arm/'status.json',dict(status='completed'));write(arm/'execution-audit.json',measurement)
    write(arm/'results.json',dict(scored=scored,passed=53,errors=7,measurement=measurement))
    if scored==82:
        overlap.check_final_review(c['pair'],c['transition'],c['audited'])
        assert read(arm/'status.json')['user_cancelled']==3
    else:
        with pytest.raises(ValueError):overlap.check_final_review(c['pair'],c['transition'],c['audited'])
