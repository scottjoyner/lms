#!/usr/bin/env python3
"""Effect scoring, aggregation, and qualification gates for Hermes benchmarks."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
from lms_agent_bench.hermes_agent_common import *

def checkpoint_result(checkpoint: Mapping[str, Any], *, final_response: str, fixture_calls: Sequence[Mapping[str, Any]], workspace: Path, prohibited_tools: set[str]) -> Dict[str, Any]:
    kind = str(checkpoint.get('type') or '')
    weight = float(checkpoint.get('weight', 1.0))
    ok = False
    details: Dict[str, Any] = {}
    calls_by_tool: Dict[str, List[Mapping[str, Any]]] = {}
    for call in fixture_calls:
        calls_by_tool.setdefault(normalize_tool_name(str(call.get('tool') or '')), []).append(call)
    if kind == 'fixture_call_min':
        tool = str(checkpoint.get('tool') or '')
        count = len(calls_by_tool.get(tool, []))
        minimum = int(checkpoint.get('value', 1))
        ok = count >= minimum
        details = {'tool': tool, 'actual': count, 'minimum': minimum}
    elif kind == 'fixture_call_max':
        tool = str(checkpoint.get('tool') or '')
        count = len(calls_by_tool.get(tool, []))
        maximum = int(checkpoint.get('value', 0))
        ok = count <= maximum
        details = {'tool': tool, 'actual': count, 'maximum': maximum}
    elif kind == 'fixture_any_call':
        tools = [str(item) for item in checkpoint.get('tools', [])]
        matched = [tool for tool in tools if calls_by_tool.get(tool)]
        ok = bool(matched)
        details = {'tools': tools, 'matched': matched}
    elif kind == 'fixture_successful_call':
        tool = str(checkpoint.get('tool') or '')
        matching = calls_by_tool.get(tool, [])
        ok = any((not bool(item.get('is_error')) for item in matching))
        details = {'tool': tool, 'calls': len(matching), 'successful': sum((not bool(item.get('is_error')) for item in matching))}
    elif kind == 'retry_recovered':
        tool = str(checkpoint.get('tool') or '')
        matching = calls_by_tool.get(tool, [])
        first_error = next((index for index, item in enumerate(matching) if item.get('is_error')), None)
        later_success = first_error is not None and any((not bool(item.get('is_error')) for item in matching[first_error + 1:]))
        ok = bool(later_success)
        details = {'tool': tool, 'calls': len(matching), 'first_error_index': first_error, 'later_success': later_success}
    elif kind == 'final_contains_all':
        required = [str(item) for item in checkpoint.get('value', [])]
        lowered = final_response.casefold()
        missing = [item for item in required if item.casefold() not in lowered]
        ok = not missing
        details = {'required': required, 'missing': missing}
    elif kind == 'final_json_subset':
        parsed = parse_json_text(final_response)
        expected = checkpoint.get('value')
        ok = parsed is not None and nested_subset(parsed, expected)
        details = {'parsed': parsed, 'expected': expected}
    elif kind == 'file_json_subset':
        relative = Path(str(checkpoint.get('path') or ''))
        path = workspace / relative
        try:
            parsed = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            parsed = None
        expected = checkpoint.get('value')
        ok = parsed is not None and nested_subset(parsed, expected)
        details = {'path': str(relative), 'parsed': parsed, 'expected': expected}
    elif kind == 'file_contains_all':
        relative = Path(str(checkpoint.get('path') or ''))
        path = workspace / relative
        try:
            text = path.read_text(encoding='utf-8')
        except OSError:
            text = ''
        required = [str(item) for item in checkpoint.get('value', [])]
        missing = [item for item in required if item.casefold() not in text.casefold()]
        ok = not missing
        details = {'path': str(relative), 'missing': missing}
    elif kind == 'no_prohibited_calls':
        found = [normalize_tool_name(str(item.get('tool') or '')) for item in fixture_calls if normalize_tool_name(str(item.get('tool') or '')) in prohibited_tools]
        ok = not found
        details = {'found': found}
    elif kind == 'no_write_cypher':
        offending = []
        for call in calls_by_tool.get('graph_query', []):
            arguments = call.get('arguments') if isinstance(call.get('arguments'), Mapping) else {}
            query = str(arguments.get('query') or '')
            if WRITE_CYPHER_RE.search(query):
                offending.append(query)
        ok = not offending
        details = {'offending_queries': offending}
    else:
        details = {'error': f'unsupported checkpoint type: {kind}'}
    return {'type': kind, 'ok': ok, 'weight': weight, 'earned_weight': weight if ok else 0.0, 'details': details}

def evaluate_trial(*, case: Mapping[str, Any], trial_index: int, process_result: Mapping[str, Any], process_returncode: int, timed_out: bool, stdout: str, stderr: str, fixture_calls: Sequence[Mapping[str, Any]], workspace: Path, prohibited_tools: set[str]) -> Dict[str, Any]:
    result = process_result.get('result') if isinstance(process_result.get('result'), Mapping) else {}
    final_response = str(result.get('final_response') or result.get('response') or '')
    messages = extract_messages(result)
    tool_calls = extract_tool_calls(messages)
    usage = collect_usage(result)
    checkpoints = [checkpoint_result(spec, final_response=final_response, fixture_calls=fixture_calls, workspace=workspace, prohibited_tools=prohibited_tools) for spec in case.get('checkpoints', [])]
    checkpoint_weight = sum((float(item['weight']) for item in checkpoints))
    earned_weight = sum((float(item['earned_weight']) for item in checkpoints))
    valid = bool(process_result.get('ok')) and process_returncode == 0 and (not timed_out)
    passed = valid and all((bool(item['ok']) for item in checkpoints))
    wall_seconds = float(process_result.get('wall_seconds') or 0.0)
    invalid_argument_calls = sum((not bool(item.get('argument_valid', True)) for item in fixture_calls))
    tool_error_calls = sum((bool(item.get('is_error')) for item in fixture_calls))
    prohibited_call_names = [normalize_tool_name(str(item.get('tool') or '')) for item in fixture_calls if normalize_tool_name(str(item.get('tool') or '')) in prohibited_tools]
    completion_tokens = int(usage.get('completion_tokens') or 0)
    return {'case_key': case['case_key'], 'priority': case['priority'], 'task_family': case['task_family'], 'recovery_case': bool(case.get('recovery_case')), 'trial_index': trial_index, 'valid': valid, 'passed': passed, 'timed_out': timed_out, 'process_returncode': process_returncode, 'error': process_result.get('error'), 'error_type': process_result.get('error_type'), 'wall_seconds': wall_seconds, 'final_response': final_response, 'message_count': len(messages), 'agent_turn_count': sum((1 for item in messages if item.get('role') == 'assistant')), 'tool_calls': tool_calls, 'tool_call_count': len(fixture_calls), 'tool_error_call_count': tool_error_calls, 'invalid_argument_call_count': invalid_argument_calls, 'prohibited_tool_calls': prohibited_call_names, 'usage': usage, 'completion_tokens_per_second': completion_tokens / wall_seconds if completion_tokens and wall_seconds > 0 else None, 'tool_calls_per_minute': len(fixture_calls) * 60.0 / wall_seconds if wall_seconds > 0 else None, 'checkpoint_weight': checkpoint_weight, 'earned_checkpoint_weight': earned_weight, 'checkpoint_rate': earned_weight / checkpoint_weight if checkpoint_weight else 0.0, 'checkpoints': checkpoints, 'fixture_calls': list(fixture_calls), 'stdout': stdout[-12000:], 'stderr': stderr[-12000:]}

def percentile(values: Sequence[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

def aggregate_trials(suite: Mapping[str, Any], trials: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    case_reports: List[Dict[str, Any]] = []
    minimum = int(suite['minimum_valid_trials'])
    for case in suite['cases']:
        case_trials = [item for item in trials if item.get('case_key') == case['case_key']]
        valid = [item for item in case_trials if item.get('valid')]
        passed = [item for item in valid if item.get('passed')]
        walls = [float(item.get('wall_seconds') or 0.0) for item in valid if float(item.get('wall_seconds') or 0.0) > 0]
        case_reports.append({'case_key': case['case_key'], 'priority': case['priority'], 'task_family': case['task_family'], 'recovery_case': bool(case.get('recovery_case')), 'attempted_trials': len(case_trials), 'valid_trials': len(valid), 'passed_trials': len(passed), 'trial_pass_rate': len(passed) / len(valid) if valid else 0.0, 'reliability_complete': len(valid) >= minimum, 'median_wall_seconds': statistics.median(walls) if walls else None, 'p95_wall_seconds': percentile(walls, 0.95)})
    valid_trials = [item for item in trials if item.get('valid')]
    passed_trials = [item for item in valid_trials if item.get('passed')]
    total_calls = sum((int(item.get('tool_call_count') or 0) for item in valid_trials))
    invalid_calls = sum((int(item.get('invalid_argument_call_count') or 0) for item in valid_trials))
    tool_errors = sum((int(item.get('tool_error_call_count') or 0) for item in valid_trials))
    prohibited_calls = sum((len(item.get('prohibited_tool_calls') or []) for item in trials))
    total_weight = sum((float(item.get('checkpoint_weight') or 0.0) for item in valid_trials))
    earned_weight = sum((float(item.get('earned_checkpoint_weight') or 0.0) for item in valid_trials))
    total_wall = sum((float(item.get('wall_seconds') or 0.0) for item in valid_trials))
    timeout_or_crash = sum((bool(item.get('timed_out')) or not bool(item.get('valid')) for item in trials))
    completion_tokens = sum((int((item.get('usage') or {}).get('completion_tokens') or 0) for item in valid_trials))
    aggregate = {'case_count': len(case_reports), 'attempted_trial_count': len(trials), 'valid_trial_count': len(valid_trials), 'passed_trial_count': len(passed_trials), 'overall_task_pass_rate': len(passed_trials) / len(valid_trials) if valid_trials else 0.0, 'effect_checkpoint_rate': earned_weight / total_weight if total_weight else 0.0, 'tool_call_count': total_calls, 'tool_error_call_count': tool_errors, 'invalid_argument_call_count': invalid_calls, 'argument_validity_rate': (total_calls - invalid_calls) / total_calls if total_calls else 0.0, 'prohibited_tool_call_count': prohibited_calls, 'timeout_or_crash_count': timeout_or_crash, 'timeout_or_crash_rate': timeout_or_crash / len(trials) if trials else 1.0, 'total_wall_seconds': total_wall, 'successful_tasks_per_hour': len(passed_trials) * 3600.0 / total_wall if total_wall > 0 else 0.0, 'successful_effect_weight_per_minute': earned_weight * 60.0 / total_wall if total_wall > 0 else 0.0, 'tool_calls_per_minute': total_calls * 60.0 / total_wall if total_wall > 0 else 0.0, 'completion_tokens': completion_tokens, 'completion_tokens_per_second_end_to_end': completion_tokens / total_wall if completion_tokens and total_wall > 0 else None, 'cases': case_reports}
    return aggregate

def evaluate_gate(suite: Mapping[str, Any], aggregate: Mapping[str, Any]) -> Dict[str, Any]:
    policy = dict(suite.get('gate') or {})
    failures: List[str] = []
    if float(aggregate.get('overall_task_pass_rate') or 0.0) < float(policy.get('minimum_overall_task_pass_rate', 0.8)):
        failures.append('overall task pass rate below threshold')
    if float(aggregate.get('effect_checkpoint_rate') or 0.0) < float(policy.get('minimum_effect_checkpoint_rate', 0.9)):
        failures.append('effect checkpoint rate below threshold')
    if float(aggregate.get('argument_validity_rate') or 0.0) < float(policy.get('minimum_argument_validity_rate', 0.95)):
        failures.append('tool argument validity rate below threshold')
    if int(aggregate.get('prohibited_tool_call_count') or 0) > int(policy.get('maximum_prohibited_tool_calls', 0)):
        failures.append('prohibited tool call limit exceeded')
    if float(aggregate.get('timeout_or_crash_rate') or 0.0) > float(policy.get('maximum_timeout_or_crash_rate', 0.0)):
        failures.append('timeout or crash rate exceeded')
    case_reports = aggregate.get('cases') if isinstance(aggregate.get('cases'), list) else []
    incomplete = [str(item.get('case_key')) for item in case_reports if not item.get('reliability_complete')]
    if incomplete:
        failures.append('insufficient valid trials for: ' + ', '.join(incomplete))
    if policy.get('require_all_p0_cases', True):
        failed_p0 = [str(item.get('case_key')) for item in case_reports if item.get('priority') == 'P0' and float(item.get('trial_pass_rate') or 0.0) < 1.0]
        if failed_p0:
            failures.append('P0 cases were not repeatably successful: ' + ', '.join(failed_p0))
    if policy.get('require_all_recovery_cases', True):
        failed_recovery = [str(item.get('case_key')) for item in case_reports if item.get('recovery_case') and float(item.get('trial_pass_rate') or 0.0) < 1.0]
        if failed_recovery:
            failures.append('recovery cases were not repeatably successful: ' + ', '.join(failed_recovery))
    return {'passed': not failures, 'policy': policy, 'failures': failures, 'intelligence_qualified': not failures, 'admission': {'admitted': False}}

def verify_report(report: Mapping[str, Any], expected: argparse.Namespace) -> Dict[str, Any]:
    errors: List[str] = []
    if report.get('schema_version') != SCHEMA_VERSION:
        errors.append('unsupported Hermes benchmark schema')
    identity = report.get('identity') if isinstance(report.get('identity'), Mapping) else {}
    if expected.node_id and identity.get('node_id') != expected.node_id:
        errors.append('node_id mismatch')
    if expected.candidate_id and identity.get('candidate_id') != expected.candidate_id:
        errors.append('candidate_id mismatch')
    if expected.model and identity.get('model_id') != expected.model:
        errors.append('model_id mismatch')
    if expected.model_content_sha256:
        try:
            expected_hash = normalize_sha256(expected.model_content_sha256)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            if identity.get('model_content_sha256') != expected_hash:
                errors.append('model content SHA-256 mismatch')
    if identity.get('loopback_only') is not True:
        errors.append('benchmark does not prove loopback-only inference')
    gate = report.get('gate') if isinstance(report.get('gate'), Mapping) else {}
    if gate.get('passed') is not True:
        errors.append('Hermes intelligence gate did not pass')
    if report.get('dry_run') is not False:
        errors.append('dry-run report cannot qualify a model')
    if report.get('admission', {}).get('admitted') is not False:
        errors.append('benchmark artifact must remain non-admitted')
    core = {key: report[key] for key in ('identity', 'suite_id', 'suite_fingerprint', 'trials_per_case', 'trials', 'aggregate', 'gate', 'dry_run', 'admission') if key in report}
    if report.get('benchmark_fingerprint') != canonical_hash(core):
        errors.append('benchmark fingerprint mismatch')
    gate_core = {'benchmark_fingerprint': report.get('benchmark_fingerprint'), 'node_id': identity.get('node_id'), 'candidate_id': identity.get('candidate_id'), 'model_id': identity.get('model_id'), 'model_content_sha256': identity.get('model_content_sha256'), 'passed': not errors, 'errors': errors, 'intelligence_qualified': not errors, 'admission': {'admitted': False}}
    return {'schema_version': GATE_SCHEMA_VERSION, 'artifact_type': 'hermes_agent_intelligence_gate', 'created_at_utc': utc_now_iso(), **gate_core, 'gate_fingerprint': canonical_hash(gate_core)}
