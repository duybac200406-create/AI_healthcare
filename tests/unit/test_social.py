from ai_healthcare.services.social import classify_social


def test_pure_social_messages_are_recognised():
    assert classify_social("xin chào") == "greeting"
    assert classify_social("Xin chào!") == "greeting"
    assert classify_social("cám ơn bạn") == "thanks"
    assert classify_social("Cảm ơn bạn nhiều nhé") == "thanks"
    assert classify_social("tạm biệt") == "farewell"
    assert classify_social("ok") == "ack"


def test_messages_with_any_other_word_are_not_social():
    assert classify_social("dạ có") is None
    assert classify_social("cảm ơn, tôi bị mệt") is None
    assert classify_social("chào bạn tôi bị mệt") is None
    assert classify_social("") is None
    assert classify_social("một câu hỏi bình thường về sức khỏe nói chung") is None
