"""Fail-closed synthetic graph owner fixture, including archived Liquid negatives."""
from __future__ import annotations

import pytest
from lms_agent_bench.hermes_mcp_fixture import Fixture, owner_lookup_query_admissible


@pytest.mark.parametrize('query,params', [
    ("MATCH (s:Service {name:'router-api'}) RETURN s.owner AS owner", {}),
    ("MATCH (s:Service) WHERE s.name=$target RETURN s.owner", {'target':'router-api'}),
    ('MATCH (s:Service) WHERE s.name="router-api" RETURN s.owner', {}),
])
def test_exact_owner_lookups_are_admissible(tmp_path, query, params):
    allowed, _ = owner_lookup_query_admissible(query,params)
    assert allowed
    fixture=Fixture(tmp_path/'state',tmp_path/'workspace','tool_distractor_selection')
    result,is_error=fixture.call('graph_query', {'query':query,'parameters':params,'read_only':True})
    assert not is_error
    assert result['records'][0]['owner']=='team-atlas'


@pytest.mark.parametrize('query,params', [
    ('MATCH (n:router-api) RETURN n.owner, n',{}),
    ('MATCH (n:router-api) RETURN n',{}),
    ('MATCH (n) RETURN n LIMIT 50',{}),
    ('SELECT * FROM services WHERE name="router-api"',{}),
    ("MATCH (s:Service {name:'worker-api'}) RETURN s.owner",{}),
    ('MATCH (s:Service) WHERE s.name=$target RETURN s.owner',{'target':'worker-api'}),
    ('MATCH (s:Service) WHERE s.name=$target RETURN s.owner',{'target':'worker-api','unused':'router-api'}),
    ("MATCH (s:Service {name:'router-api'}) DELETE s RETURN s.owner",{}),
])
def test_unattributed_or_malformed_owner_lookups_fail_closed(tmp_path, query, params):
    allowed,_=owner_lookup_query_admissible(query,params)
    assert not allowed
    fixture=Fixture(tmp_path/'state',tmp_path/'workspace','tool_distractor_selection')
    result,is_error=fixture.call('graph_query',{'query':query,'parameters':params,'read_only':True})
    assert is_error
    assert result['retryable'] is False
    assert 'error' in result
