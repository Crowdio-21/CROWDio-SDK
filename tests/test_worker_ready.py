"""worker_ready: legacy workers send only an id; v2 workers also declare what they can handle."""

import json

import pytest

from common import V2_FLAGS, V2_MESSAGES, Message, create_worker_ready_message


def test_a_legacy_worker_ready_message_is_unchanged():
    message = create_worker_ready_message("w1")
    assert message.data == {"worker_id": "w1"}


def test_a_v2_worker_declares_protocol_flags_and_messages():
    message = create_worker_ready_message(
        "phone-1", proto_version=2, supported_flags=["adaptive_planning", "problem_reports"],
        supported_messages=["probe", "msg_ack", "flags", "load", "run", "route"], worker_type="android_kotlin")
    assert message.data == {
        "worker_id": "phone-1", "proto_version": 2, "worker_type": "android_kotlin",
        "supported_flags": ["adaptive_planning", "problem_reports"],
        "supported_messages": ["probe", "msg_ack", "flags", "load", "run", "route"],
    }


def test_the_declaration_survives_the_wire():
    sent = create_worker_ready_message("p", proto_version=2, supported_flags=list(V2_FLAGS),
                                       supported_messages=list(V2_MESSAGES))
    received = Message.from_json(sent.to_json())
    assert received.data["proto_version"] == 2
    assert received.data["supported_flags"] == list(V2_FLAGS)
    assert json.loads(sent.to_json())["data"]["supported_messages"] == list(V2_MESSAGES)


@pytest.mark.parametrize("kwargs", [{"supported_flags": ["adaptive_planing"]}, {"supported_messages": ["prefech"]}])
def test_a_misspelled_name_is_rejected_instead_of_silently_declaring_nothing(kwargs):
    with pytest.raises(ValueError):
        create_worker_ready_message("p", proto_version=2, **kwargs)
